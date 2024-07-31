import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("RESOURCES/AB_NYC_2019.csv")

df['price'].plot()
plt.ylabel('Price')
plt.title('Line Plot of Prices')
plt.show()

df.plot(kind='scatter', x='price', y='number_of_reviews', color="purple")
plt.title('Scatter Plot of Price vs Number of Reviews')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(15,10))

axs[0,0].hist(df['price'], bins=20, color="lightblue")
axs[0,0].set_title("Price vs Frequency")
axs[0,0].set_xlabel("Price")
axs[0,0].set_ylabel("Frequency")

axs[0,1].hist(df['number_of_reviews'], bins=20, color="red")
axs[0,1].set_title("Number of Reviews vs Frequency")
axs[0,1].set_xlabel("Number of Reviews")
axs[0,1].set_ylabel("Frequency")

room_type = df.groupby('room_type')['price'].mean().reset_index()
axs[0,2].bar(room_type['room_type'],room_type['price'], color='lightgreen')
axs[0,2].set_title("Average Price by Room Type")
axs[0,2].set_xlabel("Room Type")
axs[0,2].set_ylabel("Price")

neighbourhoods = df['neighbourhood_group'].value_counts().reset_index()
neighbourhoods.columns = ['neighbourhood_group', 'count']
axs[1,0].bar(neighbourhoods["neighbourhood_group"], neighbourhoods["count"], color="orange")
axs[1,0].set_title("Listings by Neighbourhood")
axs[1,0].set_xlabel("Neighbourhood")
axs[1,0].set_ylabel("Number of Listings")

axs[1,1].scatter(df['price'],df['number_of_reviews'], color="blue")
axs[1,1].set_title("Price vs Reviews")
axs[1,1].set_xlabel("Price")
axs[1,1].set_ylabel("Number of Reviews")

axs[1,2].scatter(df["price"], df["reviews_per_month"], color="purple")
axs[1,2].set_title("Reviews per Month by Price")
axs[1,2].set_xlabel("Price")
axs[1,2].set_ylabel("Reviews Per Month")


plt.tight_layout()
plt.show()