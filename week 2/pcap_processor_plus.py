import scapy.all as scapy
import numpy as np
import re
import random
from urllib.parse import unquote

# 特征名称列表，用于后续输出SVM特征权重
FEATURE_NAMES = [
    "包总长度", "协议号", "时间间隔",
    "URL长度", "Body长度", "参数个数", "百分号编码次数",
    "单引号数", "双引号数", "双横线数", "井号数", "括号数", "分号数", "逗号数",
    "select次数", "union次数", "and次数", "or次数", "sleep次数", "benchmark次数",
    "concat次数", "information_schema次数"
]

def build_dataset(pcap_file, target_sqli_ratio=0.2, random_seed=42):
    """
    解析合并后的 pcap 文件，提取特征，打标签，并调整正负样本比例
    """
    print(f"正在解析 {pcap_file} ...")
    packets = scapy.rdpcap(pcap_file)
    X_all, y_all = [], []
    last_time = None
    
    # 打标签规则（书中思路：基于已知恶意特征的规则生成y）
    sqli_pattern = re.compile(r'(?i)(union.*select|sleep\s*\(|benchmark\s*\(|and\s+\d+=\d+|or\s+\d+=\d+|%27|information_schema)', re.IGNORECASE)
    
    for pkt in packets:
        pkt_len = len(pkt)
        proto = pkt.proto if hasattr(pkt, 'proto') else 0
        time_delta = float(pkt.time - last_time) if last_time else 0.0
        last_time = pkt.time
        
        url_len, body_len, param_count, pct_count = 0, 0, 0, 0
        quote1, quote2, dash, hash_c, paren, semi, comma = 0, 0, 0, 0, 0, 0, 0
        kw_counts = [0] * 8
        
        payload_str = ""
        if pkt.haslayer(scapy.Raw):
            try:
                payload_str = pkt[scapy.Raw].load.decode('utf-8', errors='ignore')
            except:
                pass
        
        # 生成标签 y
        is_sqli = 1 if sqli_pattern.search(payload_str) else 0
        y_all.append(is_sqli)
        
        # 提取特征 X
        if "HTTP/" in payload_str:
            url_match = re.search(r'(GET|POST)\s+(\S+)', payload_str, re.IGNORECASE)
            url = url_match.group(2) if url_match else ""
            body = ""
            parts = payload_str.split('\r\n\r\n', 1)
            if len(parts) > 1:
                body = parts[1]
            combined = unquote(url + " " + body)
            
            url_len = len(url)
            body_len = len(body)
            param_count = combined.count('&') + combined.count('?')
            pct_count = url.count('%')
            quote1 = combined.count("'")
            quote2 = combined.count('"')
            dash = combined.count('--')
            hash_c = combined.count('#')
            paren = combined.count('(') + combined.count(')')
            semi = combined.count(';')
            comma = combined.count(',')
            
            keywords = ['select', 'union', 'and', 'or', 'sleep', 'benchmark', 'concat', 'information_schema']
            kw_counts = [combined.lower().count(kw) for kw in keywords]
        
        features = [
            float(pkt_len), float(proto), float(time_delta),
            float(url_len), float(body_len), float(param_count), float(pct_count),
            float(quote1), float(quote2), float(dash), float(hash_c), float(paren), float(semi), float(comma)
        ] + [float(c) for c in kw_counts]
        
        X_all.append(features)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"原始数据：总样本 {len(y_all)}，攻击样本 {sum(y_all)}，正常/杂乱样本 {len(y_all)-sum(y_all)}")
    
    # 数据清洗与下采样（书中处理非平衡数据的思路）
    X_pos = X_all[y_all == 1]
    X_neg = X_all[y_all == 0]
    n_pos = len(X_pos)
    
    if n_pos == 0:
        raise ValueError("未检测到任何SQL注入样本，请检查pcap文件或打标签规则！")
        
    # 保持20%攻击 / 80%正常
    n_neg_needed = int(n_pos * (1 - target_sqli_ratio) / target_sqli_ratio)
    n_neg_needed = min(n_neg_needed, len(X_neg))
    
    random.seed(random_seed)
    sampled_neg_idx = random.sample(range(len(X_neg)), n_neg_needed)
    X_neg_sampled = X_neg[sampled_neg_idx]
    
    X = np.concatenate([X_pos, X_neg_sampled])
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg_needed, dtype=int)])
    
    # 打乱
    shuffle_idx = np.random.RandomState(random_seed).permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    print(f"下采样后：攻击样本 {sum(y)}，正常样本 {len(y)-sum(y)}，攻击占比 {sum(y)/len(y)*100:.2f}%")
    return X, y