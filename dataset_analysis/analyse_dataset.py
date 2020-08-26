import seaborn as sns
import matplotlib.pyplot as plt


def analyse(dataframe):
    df = dataframe
    # display the dimensions of the dataset
    print("dataframe dimensions: ", df.shape)

    # confirm the data types of the features
    print(df.dtypes)
    print(df.dtypes[df.dtypes == 'object'])

    # display some example observations from the data set
    print(df.head(10))
    print(df.tail())

    # distributions of categorical features
    print(df.describe(include=['object']))
    for feature in df.dtypes[df.dtypes == 'object'].index:
        sns.countplot(y=feature, data=df)
        plt.show()

