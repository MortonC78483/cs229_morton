import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# plotting the losses for paper/poster

base_3 = np.loadtxt("data/nn_baseline_0.3pct_loss.csv")
base_0= np.loadtxt("data/nn_baseline_0pct_loss.csv")
base_5 = np.loadtxt("data/nn_baseline_0.5pct_loss.csv")

cot_3 = np.loadtxt("data/nn_coteaching1_0.3pct_loss.csv")
cot_0= np.loadtxt("data/nn_coteaching1_0pct_loss.csv")
cot_5 = np.loadtxt("data/nn_coteaching1_0.5pct_loss.csv")
plt.rcParams.update({'font.size': 12})

plt.figure()
plt.plot(np.hstack((base_0, cot_0)), label = ["TN: Baseline", "FN: Baseline",
                                              "TN: Co-Teaching", "FN: Co-Teaching"])
plt.ylabel("Loss at end of each epoch")
plt.xlabel("Epoch")
plt.title("A: 0%")
leg = plt.legend()
plt.ylim(0, 1.9)
plt.gca().set_aspect(8)
plt.savefig("data/fig_A.png")


plt.figure()
plt.plot(np.hstack((base_3, cot_3)), label = ["TN: Baseline", "FN: Baseline",
                                              "TN: Co-Teaching", "FN: Co-Teaching"])
plt.ylabel("Loss at end of each epoch")
plt.xlabel("Epoch")
plt.title("B: 30%")
leg = plt.legend()
plt.ylim(0, 1.9)
plt.gca().set_aspect(8)
plt.savefig("data/fig_B.png")

plt.figure()
plt.plot(np.hstack((base_5, cot_5)), label = ["TN: Baseline", "FN: Baseline",
                                              "TN: Co-Teaching", "FN: Co-Teaching"])
plt.ylabel("Loss at end of each epoch")
plt.xlabel("Epoch")
plt.title("C: 50%")
leg = plt.legend()
plt.ylim(0, 1.9)
plt.gca().set_aspect(8)
plt.savefig("data/fig_C.png")