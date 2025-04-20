import logging
import os

import awswrangler as wr


def upload2s3(save_dir, s3_path, model_id):
    """Upload embeddings from local directory to S3

    Args:
        save_path (str): Local path where embeddings are saved
        model_id (str): Identifier for the model used
    """
    try:
        model_dir = os.path.join(save_dir, f"model_id={model_id}")

        # Upload all parquet files in model directory to S3
        for file in os.listdir(model_dir):
            if file.endswith(".parquet"):
                local_path = os.path.join(model_dir, file)
                df = wr.s3.read_parquet(local_path)
                wr.s3.to_parquet(
                    df=df,
                    path=s3_path,
                    mode="append",
                    partition_cols=[model_id],
                    dataset=True,
                    compression="snappy",
                    max_rows_by_file=1000000,
                )
                logging.info(f"Uploaded {file} to {s3_path}")

    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")
        raise
