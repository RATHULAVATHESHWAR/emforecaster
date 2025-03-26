import numpy as np
from tslearn.shapelets import LearningShapelets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X is a list of 1D numpy arrays of varying lengths
# and Y is a numpy array of labels

# Load open_neuro
from emforecaster.utils.dataloading import load_open_neuro_interchannel

data = load_open_neuro_interchannel(
    patient_cluster="ummc",
    kernel_size=24,
    kernel_stride=-1,
    window_size=100000,
    window_stride=1,
    dtype="float32",
    pool_type="avg",
    balance=True,
    scale=True,
    train_split=0.6,
    val_split=0.2,
    seed=1995,
    task="binary",
    full_channels=True,
    multicluster=False,
    resizing_mode="pad_trunc",
    median_seq_len=False,
    median_seq_only=False,
)


X_train, y_train, _, X_val, y_val, val_ch_ids, X_test, y_test, _, _ = data

# max_size = np.max([np.max([np.max(x) for x in X_train]), np.max([np.max(x) for x in X_test])])

# Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit the Learning Shapelets model
shapelets_clf = LearningShapelets(
    n_shapelets_per_size={10: 5, 20: 5},
    max_iter=100,
    batch_size=32,
    verbose=1,
    optimizer="adam",
    scale=True,
    # weight_regularizer=0.2,
    # max_size=max_size,
)
shapelets_clf.fit(X_train, y_train)

# Make predictions
y_pred = shapelets_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Transform the data to get shapelet distances
X_transformed = shapelets_clf.transform(X_test)
print(f"Transformed shape: {X_transformed.shape}")
