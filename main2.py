import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 132  # обсяг вибірки
mean = 100  # математичне сподівання
std = 2  # стандартне відхилення

# Генерувати вибірку з нормального розподілу
sample = np.random.normal(mean, std, n)
print(sample)

# Будувати полігон
x = np.linspace(0, 132, 132)
plt.plot(sample, x, ':o')
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
freq = np.cumsum(sorted_sample) / np.sum(sorted_sample)
fig, ax1 = plt.subplots()
ax1.bar(np.arange(len(sorted_sample)), sorted_sample, color='b')
ax1.set_xlabel('Значення')
ax1.set_ylabel('Частота', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(freq, 'r--')
ax2.set_ylabel('Кумулятивна частота', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Діаграма Парето')
plt.show()

# Будувати кругову діаграму
unique_values, value_counts = np.unique(sample, return_counts=True)
plt.figure(figsize=(8, 8))
plt.pie(value_counts, labels=unique_values, autopct='%1.1f%%')
plt.title('Кругова діаграма')
plt.show()

confidence_level = 0.95
alpha = 1 - confidence_level

standard_error = sample_std / np.sqrt(n)

margin_of_error = stats.t.ppf(1 - alpha / 2, n - 1) * standard_error

mean_lower = sample_mean - margin_of_error
mean_upper = sample_mean + margin_of_error

print(f"Інтервал довірчий на мат. сподівання ({confidence_level * 100}%): ({mean_lower}, {mean_upper})")

#yessssssssssssssssssssssssssssssssssssssssssssss

df = n - 1

chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

std_lower = np.sqrt((n - 1) * sample_variance / chi2_upper)
std_upper = np.sqrt((n - 1) * sample_variance / chi2_lower)

print(f"Інтервал довірчий на відхилення ({confidence_level * 100}%): ({std_lower}, {std_upper})")

#nooooooooooooooooooooooooooooooooooooooooooooooooooooo


confidence_levels = [0.8, 0.9, 0.95, 0.99]
sample_sizes = [50, 100, 150, 200]

mean_estimates = np.zeros((len(confidence_levels), len(sample_sizes)))
std_estimates = np.zeros((len(confidence_levels), len(sample_sizes)))

for i, confidence_level in enumerate(confidence_levels):
    for j, sample_size in enumerate(sample_sizes):

        new_sample = np.random.normal(mean, std, sample_size)

        new_sample_mean = np.mean(new_sample)
        new_sample_std = np.std(new_sample, ddof=1)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, sample_size - 1)
        chi2_lower = stats.chi2.ppf(alpha / 2, sample_size - 1)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, sample_size - 1)

        mean_standard_error = new_sample_std / np.sqrt(sample_size)
        mean_margin_of_error = t_critical * mean_standard_error
        std_lower = np.sqrt((sample_size - 1) * new_sample_std ** 2 / chi2_upper)
        std_upper = np.sqrt((sample_size - 1) * new_sample_std ** 2 / chi2_lower)

        mean_estimates[i, j] = new_sample_mean
        std_estimates[i, j] = new_sample_std

        print(f"Розмір: {sample_size}, Довіреність: {confidence_level}")
        print("Інтервал довірчий на мат. сподівання: ({:.4f}, {:.4f})".format(new_sample_mean - mean_margin_of_error,
                                                                  new_sample_mean + mean_margin_of_error))
        print("Інтервал довірчий на відхилення: ({:.4f}, {:.4f})".format(std_lower, std_upper))
