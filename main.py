import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 132  # обсяг вибірки
mean = 0  # математичне сподівання
std = 2  # стандартне відхилення

# Генерувати вибірку з нормального розподілу
sample = np.random.normal(mean, std, n)
print(sample)

# Будувати полігон
x = np.linspace(0, 132, 132)
plt.plot(x, sample, ':o')
plt.grid
plt.xlabel('Значення')
plt.title('Полігон')
plt.show

# Будувати гістограму частот
plt.figure(figsize=(8, 4))
plt.hist(sample, bins='auto', alpha=0.7)
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.title('Гістограма частот')
plt.show()

# Розрахувати вибіркове середнє
sample_mean = np.mean(sample)

# Розрахувати медіану
sample_median = np.median(sample)

# Розрахувати моду
sample_mode = stats.mode(sample).mode[0]

# Розрахувати вибіркову дисперсію
sample_variance = np.var(sample)

# Розрахувати середньоквадратичне відхилення
sample_std = np.std(sample)

print('Вибіркове середнє:', sample_mean)
print('Медіана:', sample_median)
print('Мода:', sample_mode)
print('Вибіркова дисперсія:', sample_variance)
print('Середньоквадратичне відхилення:', sample_std)

# Будувати діаграму розмаху
plt.figure(figsize=(8, 4))
plt.boxplot(sample)
plt.ylabel('Значення')
plt.title('Діаграма розмаху')
plt.show()

# Будувати діаграму Парето
sorted_sample = np.sort(sample)[::-1]
cumulative_percentage = np.cumsum(sorted_sample) / np.sum(sorted_sample)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(sorted_sample) + 1), cumulative_percentage, 'o-')
plt.xlabel('Номер')
plt.ylabel('Накопичена частка')
plt.title('Діаграма Парето')
plt.show()

# Будувати кругову діаграму
unique_values, value_counts = np.unique(sample, return_counts=True)
plt.figure(figsize=(8, 8))
plt.pie(value_counts, labels=unique_values, autopct='%1.1f%%')
plt.title('Кругова діаграма')
plt.show()
