import pandas as pd
from typing import Dict
from pathlib import Path
from loguru import logger

class Dataset:
    def __init__(self, dir: str):
        self._raw = self._load(dir)
        self._ids = self._generate_full_ids(self._raw)
        self._data = self._prepare_dataset(self._raw)

    def _make_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df["shop_id"].astype(str) + "-" + df["item_id"].astype(str)

    def _generate_full_ids(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cols = ["id", "shop_id", "item_id", "item_category_id"]
        items = raw["items"]
        shops = raw["shops"]
        data = pd.merge(items, shops, how="cross")
        data["id"] = self._make_id(data) 
        data = data[cols]
        return data

    def _load(self, dir) -> pd.DataFrame:
        dir = Path(dir)
        train = pd.read_csv(dir.joinpath("train.csv"), index_col=0)
        test = pd.read_csv(dir.joinpath("test.csv"))
        cats = pd.read_csv(dir.joinpath("item_categories.csv"))
        shops = pd.read_csv(dir.joinpath("shops.csv"))
        items = pd.read_csv(dir.joinpath("items.csv"))
        return {
            "train": train,
            "test": test,
            "cats": cats,
            "items": items,
            "shops": shops
        }

    def _prepare_dataset(self, raw: Dict[str, pd.DataFrame]):
        cols = [
            "date_block_num", "shop_id", "item_id", "id",
            "item_category_id", "item_cnt_month"
        ]

        train = raw["train"]
        train = train.drop(columns=["item_category_id"])
        test = raw["test"]
        items = raw["items"]

        test_block_num = max(train["date_block_num"].tolist()) + 1
        
        logger.info(f"Test date block num: {test_block_num}")

        test["date_block_num"] = test_block_num
        test["item_cnt_day"] = 0 

        data = pd.concat([train, test])
        data = pd.merge(data, items, on=["item_id"])
        data["id"] = self._make_id(data)
        key = ["date_block_num", "id", "shop_id", "item_id", "item_category_id"]
        data = data.groupby(by=key).agg(
            item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum")
        ).reset_index()
        data["item_cnt_month"] = data["item_cnt_month"].astype(int)
        data = data[cols]
        return data

    def generate(self, block_num: int, sample: float = 0.05):
        data = self.get(block_num) 
        n_sample = int(len(data) * sample)
        result = self._ids[~self._ids["id"].isin(data["id"])].sample(n_sample)
        result["item_cnt_month"] = 0
        result["date_block_num"] = block_num
        return result

    def get(self, block_from: int, block_to: int = None):
        if block_to is None:
            block_to = block_from

        cond_from = self._data["date_block_num"] >= block_from
        cond_to = self._data["date_block_num"] <= block_to 
        data = self._data[cond_from & cond_to]
        return data

if __name__ == "__main__":
    self = Dataset("../data/raw")
    data = self.get(20)
    append = self.generate(20)
    pd.concat([data, append])