import json
import os
from collections import Counter
from typing import Tuple
import fire
import numpy as np

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from alphagen_qlib.stock_data import CryptoDataLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna, DropnaProcessor, MinMaxNorm, CSZFillna

import sys
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录路径
parent_dir = os.path.dirname(current_dir)
# 将上一级目录路径添加到sys.path中
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

generation = 0


def main(
        seed: object = 0,
        instruments: object = "all",
        pool_capacity: object = 10,
) -> object:
    
    def _custom_eval(e):
        for k, v in EXPR_MAP.items():
            e = e.replace(k, v)
        return eval(e)

    def _metric(x, y, w):
        key = y[0]

        if key in cache:
            return cache[key]
        token_len = key.count('(') + key.count(')')
        if token_len > 20:
            return -1.
        
        # expr = eval(key)
        expr = _custom_eval(key)
        try:
            ic = calculator_train.calc_single_IC_ret(expr)
        except OutOfDataRangeError:
            ic = -1.
        if np.isnan(ic):
            ic = -1.
        cache[key] = ic
        return ic
        
    # 验证集和测试集计算 单个因子 的ic评价
    def try_single():
        top_key = Counter(cache).most_common(1)[0][0]
        expr = _custom_eval(top_key)
        ic_valid, ric_valid = calculator_valid.calc_single_all_ret(expr)
        ic_test, ric_test = calculator_test.calc_single_all_ret(expr)
        return {'ic_test': ic_test,
                'ic_valid': ic_valid,
                'ric_test': ric_test,
                'ric_valid': ric_valid}

    # 验证集和测试集计算 因子池中因子 的的ic评价
    def try_pool(capacity):
        pool = AlphaPool(capacity=capacity,
                        calculator=calculator_train,
                        ic_lower_bound=None)

        exprs = []
        for key in dict(Counter(cache).most_common(capacity)):
            exprs.append(_custom_eval(key))
        pool.force_load_exprs(exprs)
        pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

        ic_test, ric_test = pool.test_ensemble(calculator_test)
        ic_valid, ric_valid = pool.test_ensemble(calculator_valid)
        return {'ic_test': ic_test,
                'ic_valid': ic_valid,
                'ric_test': ric_test,
                'ric_valid': ric_valid}

    def ev():
        global generation
        generation += 1
        if pool_capacity:
            pool_res = [{'pool': pool_capacity, 'res': try_pool(pool_capacity)}]
        else:
            pool_res = [{'pool': cap, 'res': try_pool(cap)} for cap in (10, 20, 50, 100)]
        res = (
            [{'pool': 0, 'res': try_single()}] + pool_res
        )
        print(res)
        dir_ = '/Users/yangguangyu/Projects/QuantLearning/research/test_result/gplearn'
        os.makedirs(dir_, exist_ok=True)
        if generation % 2 == 0:
            with open(f'{dir_}/{generation}.json', 'w') as f:
                json.dump({'cache': cache, 'res': res}, f)

    cache = {}

    # 适应函数
    Metric = make_fitness(function=_metric, greater_is_better=True)

    # 用于进化的公式func
    funcs = [make_function(**func._asdict()) for func in generic_funcs]
    reseed_everything(seed)

    # 数据加载
    device = torch.device("cpu")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -7) / close - 1

    minmax_proc = MinMaxNorm(fit_start_time="2021-01-01", fit_end_time='2021-12-31')
    dropna_proc = DropnaProcessor()
    csfillna_proc = CSZFillna()
    learn_processors = [dropna_proc, minmax_proc]
    infer_processors = [csfillna_proc, dropna_proc, minmax_proc]

    features = list(FeatureType)
    data_train = CryptoDataLP(instrument=instruments,
                                start_time='2021-01-01',
                                end_time='2021-12-31',
                                device=device,
                                features=features,
                                processors=learn_processors,
                                for_train=True)
    print(f"train data: {data_train.data}, {data_train.data.shape}")
    data_valid = CryptoDataLP(instrument=instruments,
                                start_time='2022-01-01',
                                end_time='2022-12-31',
                                device=device,
                                features=features,
                                processors=infer_processors)
    print(f"valid data: {data_valid.data}, {data_valid.data.shape}")
    data_test = CryptoDataLP(instrument=instruments,
                                start_time='2023-01-01',
                                end_time='2023-09-01',
                                device=device,
                                features=features,
                                processors=infer_processors)
    print(f"test data: {data_test.data}, {data_test.data.shape}")

    # 数据集制作
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)


    # 常数和数据集里的作为特征
    data_features = ['$' + f.name.lower() for f in features]
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    terminals = data_features + constants
    
    X_train = np.array([terminals])
    y_train = np.array([[1]])
    print(f"X_train: {X_train}, y_train: {y_train}")

    est_gp = SymbolicRegressor(population_size=1000,
                               generations=40,
                               init_depth=(2, 6),
                               tournament_size=600,
                               stopping_criteria=1.,
                               p_crossover=0.3,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.01,
                               p_point_mutation=0.1,
                               p_point_replace=0.6,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.,
                               random_state=seed,
                               function_set=funcs,
                               metric=Metric,
                               const_range=None,
                               n_jobs=1)
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train))


def fire_helper(
        seed: Union[int, Tuple[int]],
        code: Union[str, List[str]],
        pool: int,
):
    if isinstance(seed, int):
        seed = (seed,)
    
    for _seed in seed:
        main(_seed,
             code,
             pool if pool else 10
            )


if __name__ == '__main__':
    # fire.Fire(fire_helper)
    main(0, "all", 10)
