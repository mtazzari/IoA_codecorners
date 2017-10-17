t = np.linspace(0., 100., 1000)

plt.plot(t, sin_signal(t))
plt.plot(t, sin_signal(t, omega=0.2))
# plt.plot(t, signal(t, omega=0.2, t0=-15))

plt.xlabel("Time (t)")
plt.ylabel("Signal")

plt.savefig("figure.png")