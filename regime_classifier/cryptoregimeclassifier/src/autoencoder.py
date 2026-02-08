# src/autoencoder.py
"""
CNN Autoencoder for feature dimensionality reduction
Uses 1D Convolutional layers to capture local patterns in features
Alternative to PCA that can capture non-linear relationships
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib


class CNNAutoencoder:
    """
    CNN (1D Convolutional) Autoencoder for feature dimensionality reduction
    Uses Conv1D layers to capture local patterns in feature vectors
    Alternative to PCA that can capture non-linear relationships
    """
    
    def __init__(self, input_dim, latent_dim=12, filters=[64, 32, 16], 
                 kernel_size=3, dropout_rate=0.2, random_state=42):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        latent_dim : int
            潜在空间维度（降维后的维度）
        filters : list
            CNN各层的filter数量
        kernel_size : int
            卷积核大小
        dropout_rate : float
            Dropout比率
        random_state : int
            随机种子
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        
        # 设置随机种子
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self):
        """
        构建1D CNN Autoencoder
        将1D特征向量reshape为 (input_dim, 1) 以便CNN处理
        """
        # 输入：1D特征向量，reshape为 (input_dim, 1) 用于Conv1D
        inputs = keras.Input(shape=(self.input_dim,), name='input')
        x = layers.Reshape((self.input_dim, 1), name='reshape_input')(inputs)
        
        # Encoder: 1D卷积层
        seq_length = self.input_dim
        for i, n_filters in enumerate(self.filters):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                name=f'conv_enc_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_enc_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_enc_{i}')(x)
            # 下采样（使用padding='same'保持长度，然后手动计算）
            x = layers.MaxPooling1D(pool_size=2, padding='same', name=f'pool_enc_{i}')(x)
            seq_length = (seq_length + 1) // 2  # 更新序列长度
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        flattened_dim = seq_length * self.filters[-1]
        
        # 瓶颈层（潜在表示）
        encoded = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        # Decoder: 从潜在表示重建
        x = layers.Dense(flattened_dim, activation='relu', name='dec_dense')(encoded)
        x = layers.Reshape((seq_length, self.filters[-1]), name='reshape_dec')(x)
        
        # 上采样和反卷积
        for i, n_filters in enumerate(reversed(self.filters[:-1])):
            x = layers.UpSampling1D(size=2, name=f'upsample_dec_{i}')(x)
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                name=f'conv_dec_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_dec_{i}')(x)
            seq_length = seq_length * 2  # 更新序列长度
        
        # 最后一层：恢复到原始维度
        x = layers.UpSampling1D(size=2, name='upsample_final')(x)
        seq_length = seq_length * 2
        
        # 确保输出长度匹配（裁剪或填充）
        if seq_length > self.input_dim:
            crop_size = seq_length - self.input_dim
            x = layers.Cropping1D(cropping=(0, crop_size), name='crop')(x)
        elif seq_length < self.input_dim:
            pad_size = self.input_dim - seq_length
            x = layers.ZeroPadding1D(padding=(0, pad_size), name='pad')(x)
        
        x = layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            activation='linear',
            padding='same',
            name='output_conv'
        )(x)
        
        # Reshape回原始形状
        decoded = layers.Reshape((self.input_dim,), name='reshape_output')(x)
        
        autoencoder = keras.Model(inputs, decoded, name='cnn1d_autoencoder')
        encoder = keras.Model(inputs, encoded, name='encoder')
        
        return autoencoder, encoder
    
    def fit(self, X, epochs=100, batch_size=128, validation_split=0.15, verbose=0, progress_callback=None):
        """
        训练CNN Autoencoder
        
        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
            输入特征矩阵
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        validation_split : float
            验证集比例
        verbose : int
            是否显示训练过程
        progress_callback : callable, optional
            进度回调函数，接收 (current_epoch, total_epochs, loss, val_loss) 参数
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self.autoencoder, self.encoder = self._build_model()
        
        # 编译
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # 准备回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # 添加进度回调
        if progress_callback:
            class ProgressCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    loss = logs.get('loss', 0) if logs else 0
                    val_loss = logs.get('val_loss', 0) if logs else 0
                    progress_callback(epoch + 1, epochs, loss, val_loss)
            callbacks.append(ProgressCallback())
        
        # 训练
        history = self.autoencoder.fit(
            X_scaled, X_scaled,  # 自编码器：输入=输出
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history.history
    
    def transform(self, X):
        """
        降维（类似PCA的transform）
        
        Parameters:
        -----------
        X : np.array, shape (n_samples, n_features)
            输入特征矩阵
        
        Returns:
        --------
        X_reduced : np.array, shape (n_samples, latent_dim)
            降维后的特征矩阵
        """
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)
    
    def fit_transform(self, X, epochs=100, progress_callback=None, **kwargs):
        """训练并降维"""
        self.fit(X, epochs=epochs, progress_callback=progress_callback, **kwargs)
        return self.transform(X)
    
    def save(self, filepath):
        """保存模型"""
        self.autoencoder.save(f"{filepath}_autoencoder.keras")
        self.encoder.save(f"{filepath}_encoder.keras")
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        
        metadata = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'random_state': self.random_state
        }
        import json
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        import json
        
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        model = cls(**metadata)
        model.autoencoder = keras.models.load_model(f"{filepath}_autoencoder.keras")
        model.encoder = keras.models.load_model(f"{filepath}_encoder.keras")
        model.scaler = joblib.load(f"{filepath}_scaler.joblib")
        
        return model
