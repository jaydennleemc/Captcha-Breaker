import os
import cv2
import joblib
import pathlib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_dir = pathlib.Path("../captcha/data")
abs_train_dir = train_dir.resolve()

data = []
label = []

for category in os.listdir(train_dir):
    for file_name in os.listdir(os.path.join(train_dir.resolve(), category)):
        img1 = cv2.imread(os.path.join(train_dir.resolve(), category, file_name), cv2.IMREAD_GRAYSCALE)
        # resize image to 28x28
        res1 = cv2.resize(img1, (28, 28))
        # reshape image to 1x784
        res1 = res1.reshape(-1)
        # convert data to list data
        res1 = res1.tolist()
        # append data to X_train
        data.append(res1)
        # append label to y_train
        label.append(category)

print(f'Total data: {len(data)}')
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=False)
print(f'train data: {len(X_train)}, test data: {len(X_test)}')

# define SVM model
clf = SVC(gamma=0.001, kernel='linear', C=100)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f'accuracy: {accuracy}')

# save model
joblib.dump(clf, '../data/captcha_model.pkl')
