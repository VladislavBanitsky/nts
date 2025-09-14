import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')


class TemperaturePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10  # Используем последние 10 измерений для прогноза

    def generate_synthetic_data(self, num_samples=1000):
        """Генерация синтетических данных для демонстрации"""
        np.random.seed(42)

        # Базовый тренд с сезонными колебаниями
        time = np.arange(num_samples)
        base_trend = 15 + 0.01 * time  # Медленный рост температуры
        seasonal = 8 * np.sin(2 * np.pi * time / 24)  # Суточные колебания
        noise = np.random.normal(0, 1, num_samples)  # Случайный шум

        temperatures = base_trend + seasonal + noise

        # Создаем датафрейм
        data = pd.DataFrame({
            'temperature': temperatures,
            'hour': time % 24,
            'day_sin': np.sin(2 * np.pi * time / 24),
            'day_cos': np.cos(2 * np.pi * time / 24)
        })

        return data

    def prepare_data(self, data):
        """Подготовка данных для обучения"""
        # Масштабирование данных
        scaled_data = self.scaler.fit_transform(data[['temperature', 'day_sin', 'day_cos']])

        X, y = [], []

        # Создание последовательностей
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, :])
            y.append(scaled_data[i, 0])  # Прогнозируем только температуру

        X = np.array(X)
        y = np.array(y)

        return X, y

    def build_model(self, input_shape):
        """Построение модели нейронной сети"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Обучение модели"""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        return history

    def predict_next_hour(self, recent_temperatures):
        """Прогнозирование температуры на следующий час"""
        if len(recent_temperatures) < self.sequence_length:
            raise ValueError(f"Нужно как минимум {self.sequence_length} предыдущих измерений")

        # Подготовка входных данных
        current_time = len(recent_temperatures)
        time_features = np.array([[
            np.sin(2 * np.pi * (current_time + i) / 24),
            np.cos(2 * np.pi * (current_time + i) / 24)
        ] for i in range(-self.sequence_length, 0)])

        input_data = np.column_stack([
            np.array(recent_temperatures[-self.sequence_length:]),
            time_features
        ])

        # Масштабирование
        input_scaled = self.scaler.transform(input_data)
        input_scaled = input_scaled.reshape(1, self.sequence_length, 3)

        # Прогноз
        prediction_scaled = self.model.predict(input_scaled, verbose=0)

        # Обратное масштабирование
        dummy = np.zeros((1, 3))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction = self.scaler.inverse_transform(dummy)[0, 0]

        return prediction

    def evaluate_model(self, X_test, y_test):
        """Оценка качества модели"""
        if self.model is None:
            raise ValueError("Модель не обучена")

        predictions_scaled = self.model.predict(X_test, verbose=0)

        # Обратное масштабирование предсказаний
        dummy = np.zeros((len(predictions_scaled), 3))
        dummy[:, 0] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy)[:, 0]

        # Обратное масштабирование реальных значений
        dummy[:, 0] = y_test
        actual = self.scaler.inverse_transform(dummy)[:, 0]

        mae = np.mean(np.abs(predictions - actual))
        mse = np.mean((predictions - actual) ** 2)

        return predictions, actual, mae, mse


# Пример использования
def main():
    # Создаем экземпляр предсказателя
    predictor = TemperaturePredictor()

    # Генерируем синтетические данные
    print("Генерация данных...")
    data = predictor.generate_synthetic_data(2000)

    # Подготовка данных
    print("Подготовка данных...")
    X, y = predictor.prepare_data(data)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Обучение модели
    print("Обучение модели...")
    history = predictor.train(X_train, y_train, epochs=30)

    # Оценка модели
    print("Оценка модели...")
    predictions, actual, mae, mse = predictor.evaluate_model(X_test, y_test)

    print(f"\nРезультаты оценки:")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}°C")
    print(f"Средняя квадратичная ошибка (MSE): {mse:.2f}°C²")

    # Демонстрация прогноза
    print("\nДемонстрация прогноза:")
    recent_temps = data['temperature'].values[-predictor.sequence_length:]
    next_hour_temp = predictor.predict_next_hour(recent_temps.tolist())

    print(f"Последние температуры: {recent_temps[-5:].round(1)}°C")
    print(f"Прогноз на следующий час: {next_hour_temp:.1f}°C")

    # Визуализация
    plt.figure(figsize=(15, 5))

    # График обучения
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Ошибка обучения')
    plt.plot(history.history['val_loss'], label='Ошибка валидации')
    plt.title('История обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()

    # График прогнозов vs реальные значения
    plt.subplot(1, 2, 2)
    plt.plot(actual[:100], label='Реальные значения', alpha=0.7)
    plt.plot(predictions[:100], label='Прогнозы', alpha=0.7)
    plt.title('Сравнение прогнозов и реальных значений')
    plt.xlabel('Время')
    plt.ylabel('Температура (°C)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return predictor


# Сохранение и загрузка модели
def save_model(predictor, filename='temperature_predictor.h5'):
    predictor.model.save(filename)
    print(f"Модель сохранена как {filename}")


if __name__ == "__main__":
    # Запуск обучения и демонстрации
    model = main()

    # Пример использования обученной модели
    print("\n" + "=" * 50)
    print("Пример использования в реальном времени:")

    # Симуляция получения новых данных
    current_temps = [16, 16, 16, 15, 14, 14, 13, 13, 13, 14]

    true_future_temp = 16

    try:
        prediction = model.predict_next_hour(current_temps)
        print(f"Текущая температура: {current_temps[-1]:.1f}°C")
        print(f"Прогноз на следующий час: {prediction:.1f}°C")
        print(f"Реальная температура на следующий час: {true_future_temp:.1f}°C")
    except ValueError as e:
        print(f"Ошибка: {e}")