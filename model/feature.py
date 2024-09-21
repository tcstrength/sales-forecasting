import time
import pandas as pd
from loguru import logger
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from model.dataset import Dataset

class FeatureExtraction:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._encoders = self._prepare_encoders()

    def _prepare_encoders(self):
        x = self._dataset._raw["cats"]
        x["category"] = x["item_category_id"]
        cat_encoder = pd.get_dummies(x, columns=["category"])

        x = self._dataset._raw["shops"]
        x["shop"] = x["shop_id"]
        shop_encoder = pd.get_dummies(x, columns=["shop"])

        x = pd.DataFrame(range(12), columns=["month"])
        x["mnt"] = x["month"]
        month_encoder = pd.get_dummies(x, columns=["mnt"])
        return {
            "cat_encoder": cat_encoder,
            "shop_encoder": shop_encoder,
            "month_encoder": month_encoder
        }

    def add_category(self, data: pd.DataFrame, drop_origin: bool = True):
        shop_encoder = self._encoders["shop_encoder"]
        cat_encoder = self._encoders["cat_encoder"]
        data = pd.merge(data, cat_encoder, on=["item_category_id"])
        data = pd.merge(data, shop_encoder, on=["shop_id"])
        if drop_origin:
            data = data.drop(columns=["item_category_id", "shop_id"])
        return data
    
    def add_status(self, data: pd.DataFrame, block_num: int):
        past_ids = self._dataset.get(0, block_num - 1)["id"].unique()
        data["is_new_id"] = ~data["id"].isin(past_ids)
        return data

    def add_time_based(self, data: pd.DataFrame, block_num: int):
        days_in_month = {
            0: 31,  # January
            1: 28,  # February (non-leap year)
            2: 31,  # March
            3: 30,  # April
            4: 31,  # May
            5: 30,  # June
            6: 31,  # July
            7: 31,  # August
            8: 30,  # September
            9: 31,  # October
            10: 30, # November
            11: 31  # December
        }
        month = block_num % 12
        month_enc = self._encoders["month_encoder"]
        data["month"] = month
        data["days_in_month"] = days_in_month[month]
        data = pd.merge(data, month_enc, on=["month"]) 
        data = data.drop(columns=["month"])
        return data

    def add_tsfresh(
        self, data: pd.DataFrame,
        block_num: int,
        neg_from: int,
        neg_to: int
    ):
        if neg_to <= 0:
            msg = (
                "Argument `neg_to` must be greater than or equal to 1",
                f"given neg_to={neg_to}"
            )
            raise ValueError(msg)

        if neg_from < neg_to:
            msg = (
                "Argument `neg_from` must be greater than `neg_to`, "
                f"given neg_from={neg_from}, neg_to={neg_to}"
            )
            raise ValueError(msg)

        block_from = block_num - neg_from
        block_to = block_num - neg_to
        feat = self._dataset.get(block_from, block_to)
        feat = feat[["date_block_num", "id", "item_cnt_month"]]
        suffix = f'_{neg_from}_{neg_to}'
        settings = MinimalFCParameters()

        out = extract_features(
            feat, column_id='id',
            column_sort='date_block_num',
            default_fc_parameters=settings,
            n_jobs=8
        )
        cols = {c: f'{c}{suffix}' for c in out.columns}
        out = out.rename(columns=cols)
        out = out.reset_index(names=['id'])

        data = pd.merge(data, out, on='id', how='left')
        data = data.fillna(0)
        return data

    def add_features(self, data: pd.DataFrame, block_num: int):
        data = self.add_category(data)
        data = self.add_time_based(data, block_num)
        data = self.add_tsfresh(data, block_num, 9, 6)
        data = self.add_tsfresh(data, block_num, 6, 3)
        data = self.add_tsfresh(data, block_num, 3, 2)
        data = self.add_tsfresh(data, block_num, 1, 1)
        return data

    def extract_features(self, block_num: int, include_no_sales: bool = True):
        start_time = time.time()
        logger.info(f"Extract features for block_num={block_num}")

        data = self._dataset.get(block_num)
        if include_no_sales:
            no_sales = self._dataset.generate(block_num)
            data = pd.concat([data, no_sales])
        data = self.add_features(data, block_num)
        logger.info(f"=========== DONE ({time.time() - start_time} s) ===========")
        return data

if __name__ == "__main__":
    dataset = Dataset("../data/raw")
    self = FeatureExtraction(dataset)
    data = self.extract_features(34, False)
    data
    # data = self.extract_features(34)
    # data.head(10)
    # data = self.add_tsfresh_feats(data, 33, 12, 1)
    # data[data["id"] == "10-10200"]

    # tmp = self.get_data(32, include_no_sales=False)
    # tmp[tmp["id"] == "10-10200"]

    # tmp = self.get_data(31, include_no_sales=False)
    # tmp[tmp["id"] == "10-10200"]
