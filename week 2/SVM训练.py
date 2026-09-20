import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report
from pcap_processor_plus import build_dataset, FEATURE_NAMES

if __name__ == '__main__':
    # ⚠️ 根据你的目录结构，合并后的文件叫 sql.pcap
    PCAP_FILE = "sql.pcap" 
    
    # 1. 读取数据与特征工程（书中思路：数据清洗 -> 特征化）
    X, y = build_dataset(PCAP_FILE, target_sqli_ratio=0.2)
    
    # 2. 构建模型（书中思路：实例化算法）
    # SVM必须配合标准化，否则特征量纲差异会毁掉模型
    svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, dual='auto', max_iter=5000)) 
    # dual='auto' 会自动选择最优算法
    # max_iter=5000 防止迭代次数不够报警告
    ])
    
    # 3. 划分方式：70%训练 / 30%测试，迭代10次
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7, random_state=42)
    
    # 4. 效果验证（书中思路：十折验证，此处为10轮70/30验证）
    print("\n" + "="*50)
    print("SVM (线性核) - 10轮 70/30 训练测试验证")
    print("="*50)
    
    scores = cross_val_score(svm_clf, X, y, cv=sss, n_jobs=-1)
    print(f"10轮准确率矩阵:\n{scores}")
    print(f"平均准确率: {scores.mean() * 100:.2f}%")
    
    # 5. 输出详细分类报告（取第一轮切分）
    print("\n" + "="*50)
    print("详细分类报告 (第一轮 70/30 切分)")
    print("="*50)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        break
    
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    
    print("\n--- SVM 分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=["正常/杂乱", "SQL注入"], zero_division=0))
    
    # 6. 特征权重排名（SVM线性核优势，对应书中输出结果）
    svm_model = svm_clf.named_steps['svm']
    print("\n" + "="*50)
    print("SVM 特征权重排名 (绝对值 Top 10)")
    print("="*50)
    importances = np.abs(svm_model.coef_[0])
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(FEATURE_NAMES))):
        print(f"{i+1}. {FEATURE_NAMES[indices[i]]}: {importances[indices[i]]:.4f}")