incorrect_test = np.where(y_pred_test != np.argmax(y_test, axis=1))[0]

for i in range(5):
    idx = incorrect_test[i]
    plt.imshow(x_test[idx], cmap='gray')
    true_label = np.argmax(y_test[idx])
    pred_label = y_pred_test[idx]
    plt.title(f'True: {true_label} Pred: {pred_label}')
    plt.axis('off')
    plt.show()
