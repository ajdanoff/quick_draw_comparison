# QuickDraw: сравнительный анализ моделей

[![Python](https://img.shields.io/badge/Python-3.8+-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

Небольшой экспериментальный проект по сравнению моделей для классификации изображений на датасете `quickdraw_bitmap`.  
В ноутбуке сопоставляются NumPy MLP, PyTorch MLP и CNN, а также анализируется влияние функции активации на качество обучения и инференса.

## Основные идеи

- Сравнение полносвязных и сверточных архитектур.
- Проверка ReLU и Tanh.
- Анализ качества на validation/test.
- Сравнение поведения моделей на примерах инференса.

## Данные

Используется `quickdraw_bitmap` из TensorFlow Datasets.

Кратко:
- 345 классов;
- изображения размером 28×28×1;
- многоклассовая классификация;
- загрузка через `tfds.load(...)`.

## Результаты

### Сводная таблица

| Модель | Активация | Best epoch | Best val_acc | Test acc |
|---|---:|---:|---:|---:|
| NumPy MLP | ReLU | 18 | 0.3128 | 0.3078 |
| NumPy MLP | Tanh | 18 | 0.3128 | 0.2827 |
| TorchMLP | ReLU | 10 | 0.3040 | 0.2935 |
| TorchMLP | Tanh | 7 | 0.2735 | 0.2708 |
| TorchCNN | ReLU | 10 | 0.4525 | 0.4488 |

### Инференс

| Модель | Correct / 10 |
|---|---:|
| NumPy MLP | 4 |
| Torch MLP | 3 |
| Torch CNN | 7 |

CNN показала лучший баланс качества и устойчивости, а полносвязные модели чаще ошибались на визуально похожих классах.

## Выводы

- ReLU в среднем работала лучше, чем Tanh.
- NumPy и PyTorch MLP дали близкие результаты.
- CNN оказалась сильнее на этой задаче.
- Для изображений свёрточная архитектура наиболее подходящая.

## Что можно улучшить

1. Добавить data augmentation.
2. Усилить CNN через BatchNorm, дополнительный сверточный блок, dropout и weight decay.
3. Провести анализ ошибок по классам и при необходимости использовать балансировку классов.

## Запуск

```bash
pip install tensorflow tensorflow-datasets torch numpy pandas matplotlib
jupyter notebook project_notebook.ipynb
```

## Структура

```bash
project/
├── project_notebook.ipynb
├── README.md
└── requirements.txt
```

## License

MIT License

## Contact

Alexander Zhdanov  
[alexander.jdanoff@gmail.com](mailto:alexander.jdanoff@gmail.com)