from typing import Optional, TypeVar, Callable, Optional
import os
import pickle
import warnings
import pandas as pd
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy
from datetime import datetime

_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float

class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = None,
        return_report: bool = False
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=1,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result: dict = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", str(result))

        print(report)
        print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )

if __name__ == "__main__":
    qlib_backtest = QlibBacktest()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    """ Example code
    expr = Mul(EMA(Sub(Delta(Mul(Log(open_),Constant(-30.0)),50),Constant(-0.01)),40),Mul(Div(Abs(EMA(low,50)),close),Constant(0.01)))
    data_df = data.make_dataframe(expr.evaluate(data))
    qlib_backtest.run(data_df)
    """

    """ hs300"""
    code = "hs300"
    file_path = "/Users/yangguangyu/Projects/QuantLearning/research/test_result/alphagen/new_hs300_200_5_20231230134354/30720_steps_pool.json"


    """ zz500
    code = "zz500"
    file_path = "/Users/yangguangyu/Projects/QuantLearning/research/test_result/alphagen/new_zz500_200_5_20231231190942/30720_steps_pool.json"
    """

    data = StockData(instrument=code,
                     start_time='2020-01-01',
                     end_time='2021-12-31',
                     device=torch.device("cpu"))


    expr_map = {
        "$high": "high",
        "$low": "low",
        "$volume": "volume",
        "$open": "open_",
        "$close": "close",
        "$vwap": "vwap",
        "$target": "target"
    }

    def load_alpha_exprs(file_path):
        import json
        import copy
        with open(file_path, "r") as f:
            rslt = json.load(f)
            exprs = rslt.pop("exprs")
            _exprs = []
            for expr in exprs:
                e = copy.deepcopy(expr)
                for k, v in expr_map.items():
                    e = e.replace(k, v)
                print(e)
                _exprs.append(eval(e))
            rslt["exprs"] = _exprs
            return rslt

    rslt = load_alpha_exprs(file_path)
    exprs = rslt["exprs"]
    weights = rslt["weights"]

    dfs = []
    for expr in exprs:
        e = expr.evaluate(data)
        _data_df = data.make_dataframe(e)
        dfs.append(_data_df)

    data_df = pd.concat(dfs, axis=1)
    weighted_sum = data_df.mul(weights).sum(axis=1)
    data_df['Weighted_Sum'] = weighted_sum
    qlib_backtest.run(data_df, f"/Users/yangguangyu/Projects/QuantLearning/research/test_result/alphagen/new_zz500_200_5_20231231190942/{code}_{timestamp}")
