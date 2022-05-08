import matplotlib.pyplot as plt
from processing_data.normalize_content import tag_freq_dict

sorted_tags = sorted(tag_freq_dict.items(), key=lambda x: x[1], reverse=True)[:25]
x = []
y = []
for i in range(len(sorted_tags)):
    x.append(sorted_tags[i][0])
    y.append(sorted_tags[i][1])

plt.bar(x, y)
plt.xticks(rotation='vertical')
plt.xlabel("Tags", labelpad=10)
plt.ylabel("Sum of tags by 1m questions", labelpad=15)
plt.title("Popularity of tags on stackoverflow")
plt.legend()
plt.show()
