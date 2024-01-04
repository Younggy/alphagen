import json
import os
from typing import Optional, Tuple
from datetime import datetime
import fire
from qlib.data import D
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet, TransformerSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.stock_data import StockData, StockDataLP
from alphagen_qlib.calculator import QLibStockDataCalculator
from sklearn.preprocessing import StandardScaler
import pandas as pd
from qlib.data.dataset.processor import fetch_df_by_index, get_group_columns
from qlib.data.dataset.processor import ZScoreNorm, Fillna, DropnaProcessor, MinMaxNorm, CSZFillna
import joblib
LOGDIR = '/Users/yangguangyu/Projects/QuantLearning/research/test_result/alphagen'



class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 valid_calculator: AlphaCalculator,
                 test_calculator: AlphaCalculator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)  # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
        seed: object = 0,
        instruments: object = "csi300",
        pool_capacity: object = 10,
        steps: object = 200_000
) -> object:
    reseed_everything(seed)

    # device = torch.device('cuda:0')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    minmax_proc = MinMaxNorm(fit_start_time="2015-01-01", fit_end_time='2018-12-31')
    dropna_proc = DropnaProcessor()
    csfillna_proc = CSZFillna()
    learn_processors = [dropna_proc, minmax_proc]
    infer_processors = [csfillna_proc, dropna_proc, minmax_proc]

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockDataLP(instrument=instruments,
                             start_time='2015-01-01',
                             end_time='2018-12-31',
                             device=device,
                             processors=learn_processors,
                             for_train=True)
    print(f"train data: {data_train.data}, {data_train.data.shape}")
    data_valid = StockDataLP(instrument=instruments,
                             start_time='2019-01-01',
                             end_time='2019-12-31',
                             device=device,
                             processors=infer_processors)
    print(f"valid data: {data_valid.data}, {data_valid.data.shape}")
    data_test = StockDataLP(instrument=instruments,
                            start_time='2020-01-01',
                            end_time='2023-06-30',
                            device=device,
                            processors=infer_processors)
    print(f"test data: {data_test.data}, {data_test.data.shape}")
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=100,
        show_freq=100,
        save_path=LOGDIR,
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=2,
    )

    # model = MaskablePPO(
    #     'MlpPolicy',
    #     env,
    #     policy_kwargs=dict(
    #         features_extractor_class=LSTMSharedNet,
    #         features_extractor_kwargs=dict(
    #             n_layers=2,
    #             d_model=128,
    #             dropout=0.1,
    #             device=device,
    #         ),
    #     ),
    #     gamma=1.,
    #     ent_coef=0.01,
    #     batch_size=128,
    #     tensorboard_log=LOGDIR,
    #     device=device,
    #     verbose=1,
    # )
    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=TransformerSharedNet,
            # features_extractor_kwargs=dict(
            #     n_layers=2,
            #     d_model=128,
            #     dropout=0.1,
            #     device=device,
            # ),
            features_extractor_kwargs=dict(
                n_encoder_layers=6,
                d_model=128,
                n_head=4,
                d_ffn=2048,
                dropout=0.1,
                device=device
            ),
        ),
        n_steps=2048,
        n_epochs=10,
        batch_size=256,
        gamma=1,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=0.001,
        tensorboard_log=LOGDIR,
        device=device,
        verbose=2,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )

    joblib.dump(minmax_proc, f'{name_prefix}_{timestamp}_minmax.pkl')


def fire_helper(
        seed: Union[int, Tuple[int]],
        code: Union[str, List[str]],
        pool: int,
        step: int = None
):
    if isinstance(seed, int):
        seed = (seed,)
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)
