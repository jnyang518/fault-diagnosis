import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 1, 3, 5, 4]

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='-')
plt.title('随便的折线图')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.savefig('line_plot.png')
print('Saved line_plot.png')
