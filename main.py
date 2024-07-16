import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier

class SGDClassifier:
    def __init__(self, learning_rate=0.01, n_iter=1000, penalty='l2', alpha=0.0001, loss='log', verbose=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.penalty = penalty
        self.alpha = alpha
        self.loss = loss
        self.verbose = verbose
        
    def _initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def _compute_loss(self, X, y):
        if self.loss == 'hinge':
            margins = 1 - y * (np.dot(X, self.weights) + self.bias)
            hinge_loss = np.where(margins > 0, margins, 0)
            return np.mean(hinge_loss) + self._regularization_term()
        elif self.loss == 'log':
            linear_output = np.dot(X, self.weights) + self.bias
            log_loss = np.log(1 + np.exp(-y * linear_output))
            return np.mean(log_loss) + self._regularization_term()
    
    def _regularization_term(self):
        if self.penalty == 'l2':
            return self.alpha * np.dot(self.weights, self.weights) / 2
        elif self.penalty == 'l1':
            return self.alpha * np.sum(np.abs(self.weights))
        else:
            return 0
    
    def _compute_gradient(self, X, y):
        if self.loss == 'hinge':
            margins = 1 - y * (np.dot(X, self.weights) + self.bias)
            mask = margins > 0
            dW = -np.dot(X[mask].T, y[mask]) / len(y) + self.alpha * self.weights
            db = -np.sum(y[mask]) / len(y)
        elif self.loss == 'log':
            linear_output = np.dot(X, self.weights) + self.bias
            probs = 1 / (1 + np.exp(-linear_output))
            errors = probs - (y == 1)
            dW = np.dot(X.T, errors) / len(y) + self.alpha * self.weights
            db = np.sum(errors) / len(y)
        return dW, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        for i in range(self.n_iter):
            dW, db = self._compute_gradient(X, y)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            if self.verbose and i % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Iteration {i}: Loss = {loss}")
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
    
    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear_output))

class MeanTeacherSGDClassifier:
    def __init__(self, learning_rate=0.01, n_iter=1000, alpha=0.0001, ema_decay=0.99, loss='log', verbose=False, consistency_threshold=0.9, alpha_weight=0.5):
        self.student = SGDClassifier(learning_rate=learning_rate, n_iter=n_iter, penalty='l2', alpha=alpha, loss=loss, verbose=verbose)
        self.teacher = SGDClassifier(learning_rate=learning_rate, n_iter=n_iter, penalty='l2', alpha=alpha, loss=loss, verbose=False)
        self.ema_decay = ema_decay
        self.loss = loss
        self.verbose = verbose
        self.consistency_threshold = consistency_threshold
        self.alpha_weight = alpha_weight

    def _update_teacher(self):
        self.teacher.weights = self.ema_decay * self.teacher.weights + (1 - self.ema_decay) * self.student.weights
        self.teacher.bias = self.ema_decay * self.teacher.bias + (1 - self.ema_decay) * self.student.bias

    def _consistency_loss(self, X_unlabeled):
        teacher_preds = self.teacher.predict_proba(X_unlabeled)
        student_preds = self.student.predict_proba(X_unlabeled)
        
        mask = np.max(teacher_preds, axis=1) > self.consistency_threshold
        consistency_loss = np.mean((teacher_preds[mask] - student_preds[mask]) ** 2)
        return consistency_loss

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        n_samples, n_features = X_labeled.shape
        self.student._initialize_weights(n_features)
        self.teacher._initialize_weights(n_features)
        
        for i in range(self.student.n_iter):
            # Update student
            dW_student, db_student = self.student._compute_gradient(X_labeled, y_labeled)
            self.student.weights -= self.student.learning_rate * dW_student
            self.student.bias -= self.student.learning_rate * db_student

            # Update teacher
            self._update_teacher()

            # Compute student loss
            student_loss = self.student._compute_loss(X_labeled, y_labeled)

            # Compute consistency loss
            consistency_loss = self._consistency_loss(X_unlabeled)

            # Combine losses
            combined_loss = self.alpha_weight * student_loss + (1 - self.alpha_weight) * consistency_loss

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Combined Loss = {combined_loss}")

    def predict(self, X):
        return self.student.predict(X)

    def predict_proba(self, X):
        return self.student.predict_proba(X)

# Generate datasets
X_labeled, y_labeled = make_classification(n_samples=1000, n_features=20, random_state=42)
X_unlabeled, _ = make_classification(n_samples=1000, n_features=20, random_state=24)
y_labeled = np.where(y_labeled == 0, -1, 1)  # Convert labels to -1 and 1 for our implementation

# Split the labeled dataset
X_labeled_train, X_labeled_test, y_labeled_train, y_labeled_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Standardize features
scaler_labeled = StandardScaler()
X_labeled_train = scaler_labeled.fit_transform(X_labeled_train)
X_labeled_test = scaler_labeled.transform(X_labeled_test)

scaler_unlabeled = StandardScaler()
X_unlabeled = scaler_unlabeled.fit_transform(X_unlabeled)

# Train MeanTeacherSGDClassifier
mean_teacher_sgd = MeanTeacherSGDClassifier(learning_rate=0.01, n_iter=1000, loss='log', verbose=True)
mean_teacher_sgd.fit(X_labeled_train, y_labeled_train, X_unlabeled)
mean_teacher_predictions = mean_teacher_sgd.predict(X_labeled_test)
mean_teacher_accuracy = accuracy_score(y_labeled_test, mean_teacher_predictions)

mean_teacher_accuracy
