import pandas as pd

# Load datasets
bank_df = pd.read_csv('bank_transaction_dataset.csv')
credit_card_df = pd.read_csv('credit_card_transaction_dataset.csv')

# Rename columns in bank_df
bank_df.rename(columns={'amount_transferred': 'transferred_amount',
                        'mode_of_transaction': 'transaction_mode',
                        'date_and_time': 'transaction_date_time',
                        'place_of_transaction': 'merchant',
                        'account_number': 'identity'}, inplace=True)

# Rename columns in credit_card_df and set 'transaction_mode' to 'card'
credit_card_df.rename(columns={'transaction_amount': 'transferred_amount'}, inplace=True)
credit_card_df['transaction_mode'] = 'card'

# Merge account_number and card_number into a new column 'identity'
bank_df['identity'] = bank_df['identity'].astype(str)
credit_card_df['card_number'] = credit_card_df['card_number'].astype(str)
credit_card_df['identity'] = credit_card_df['card_number']
credit_card_df.drop(columns=['card_number'], inplace=True)

# Set 'merchant' to 'unknown' for bank_df
bank_df['merchant'] = 'unknown'

# Concatenate the two datasets
merged_df = pd.concat([bank_df, credit_card_df], ignore_index=True)

# Shuffle the rows
merged_df = merged_df.sample(frac=1).reset_index(drop=True)

# Reorder columns
merged_df = merged_df[['transferred_amount', 'transaction_mode', 'transaction_date_time', 
                       'identity', 'merchant', 'label', 'amount_before_transaction', 
                       'amount_after_transaction']]

# Save the merged dataset
merged_df.to_csv('merged_transaction_dataset.csv', index=False)
