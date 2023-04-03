import pandas as pd
import sys

groundtruth_path = sys.argv[1]
prediction_path = sys.argv[2]

print(groundtruth_path, prediction_path)

#the following code chekck your submission format
df = pd.read_csv(prediction_path, encoding='utf-8')
if len(df) != 13697:
    print('number of rows does not match the raw input file, please double-check your submission')
    quit()

if 'prediction' in df.columns:
    df['predictions'] = df['prediction']

for key in ['language','predictions']:
    if key not in df.columns:
        print('%s not found in your csv file, please make sure you keep all the columns in the original input file and add your predictions to the \"predictions\" column'%key)
        quit()
print('prediction data format correct!')


truth = pd.read_csv(groundtruth_path)

if 'text' not in df.columns:
    print(f, 'Warning: text column not in prediction file')
else:
    # sort the dataframe by text to make sure the rows are aligned
    df = df.sort_values('text')
    truth = truth.sort_values('text')

df['language'] = truth['language']
df['label'] = truth['label']
df['predictions'] = df['predictions'].astype(float)
# if the label is 0, the corresponding text is not really in the test set so we are removing them 
# here before the final calculation

df = df[(df['label']!=0)]
for language in ['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese','Hindi', 'Dutch', 'Korean', 'Arabic']:
    t_df = df[df['language']==language]
    print(language, t_df['predictions'].corr(t_df['label']))
print('Overall', df['predictions'].corr(df['label']))

s_df = df[df['language'].isin(['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese'])]
print('seen_languages',s_df['predictions'].corr(s_df['label']))
u_df = df[df['language'].isin(['Hindi', 'Dutch', 'Korean', 'Arabic'])]
print('unseen_languages',u_df['predictions'].corr(u_df['label']))