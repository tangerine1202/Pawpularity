import logging

log = logging.getLogger(__name__)


def log_data_info(train_df, val_df, test_df):
    # Print dataset sizes
    log.debug(f'Training data: {len(train_df)} samples')
    log.debug(f'Validation data: {len(val_df)} samples')
    log.debug(f'Test data: {len(test_df)} samples')

    # Check feature information
    if len(train_df) > 0:
        feature_cols = [col for col in train_df.columns if col != 'image_path']
        msg = f'Features available:\n' + ', '.join(feature_cols)
        log.debug(msg)

    log.debug(f'Description of training data:\n{train_df.describe()}')
    log.debug(f'Description of validation data:\n{val_df.describe()}')
