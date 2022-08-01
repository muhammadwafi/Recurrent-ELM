x = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for x, y in enumerate(zip(x, range(5))):
    print(f"{x}-{y}")
