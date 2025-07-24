import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit,
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QStatusBar, QProgressBar, QAction, QToolBar, QButtonGroup, 
    QRadioButton, QCheckBox, QGroupBox, QSplitter, QTextBrowser,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPalette, QColor, QTextDocument, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from Detector.Detector import Detector
from DataProcessing import CleanData, HandleMissingValues, DetectAndRemoveOutliers, NormalizeData, StandardizeData
from Detector.textDetector import TextDetector
from TextProcessing import RemoveHTMLTags, RemoveSpecialChars, HandleNumbers, TokenizeText, RemoveStopwords, NormalizeText
import matplotlib.pyplot as plt
import json
import nltk
from PyQt5.QtGui import QIcon

class TextProcessor:
    def __init__(self):
        self.operations = []
    
    def add_operation(self, operation_type, **params):
        self.operations.append({
            'type': operation_type,
            'params': params
        })
    
    def execute(self, input_text):
        current_text = input_text
        for operation in self.operations:
            try:
                if operation['type'] == 'remove_html':
                    current_text = RemoveHTMLTags(current_text).run()
                elif operation['type'] == 'remove_special_chars':
                    current_text = RemoveSpecialChars(current_text).run()
                elif operation['type'] == 'handle_numbers':
                    current_text = HandleNumbers(current_text, 
                                              strategy=operation['params'].get('strategy', 'keep')).run()
                elif operation['type'] == 'tokenize':
                    token_type_val = 'word' if operation['params'].get('token_type', 'По словам') == 'По словам' else 'sentence'
                    current_text = TokenizeText(current_text, 
                                             token_type=token_type_val,
                                             lang=operation['params']['lang']).run()
                elif operation['type'] == 'remove_stopwords':
                    current_text = RemoveStopwords(current_text,
                                                lang=operation['params']['lang']).run()
                elif operation['type'] == 'normalize':
                    current_text = NormalizeText(current_text,
                                              method=operation['params'].get('method', 'stem'),
                                              lang=operation['params']['lang']).run()
            except Exception as e:
                print(f"Error in {operation['type']}: {str(e)}")
        return current_text

class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('app_icon.png'))
        self.current_data = None
        self.current_text = None
        self.history = []
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        # Главный виджет и layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Панель управления
        control_panel = QHBoxLayout()
        self.btn_load = QPushButton("📊Загрузить табличные данные (*.csv *.xlsx *.json *.parquet)")
        self.btn_load_text = QPushButton("📄Загрузить текстовые данные (*.txt *.csv *.xlsx *.json)")
        self.btn_save = QPushButton("Сохранить результат")
        self.btn_save.setEnabled(False)
        
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_load_text)
        control_panel.addWidget(self.btn_save)
        
        # Табы
        self.tabs = QTabWidget()
        
        # Вкладка данных
        self.init_data_tab()
        
        # Вкладка анализа
        self.init_analysis_tab()
        
        # Вкладка очистки
        self.init_clean_tab()
        
        # Вкладка пропусков
        self.init_missing_tab()
        
        # Вкладка выбросов
        self.init_outliers_tab()
        
        # Вкладка масштабирования
        self.init_scaling_tab()
        
        # Вкладка текстовой обработки
        self.init_text_processing_tab()
        
        # Лог операций
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        
        # Сборка главного layout
        main_layout.addLayout(control_panel)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(QLabel("Лог операций:"))
        main_layout.addWidget(self.log)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Статус бар
        self.init_status_bar()
        
        # Панель инструментов
        self.init_toolbar()

        self.init_result_tab()
    
    def init_data_tab(self):
        self.tab_data = QWidget()
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        data_layout = QVBoxLayout()
        data_layout.addWidget(self.data_table)
        self.tab_data.setLayout(data_layout)
        self.tabs.addTab(self.tab_data, "Данные")
    
    def init_analysis_tab(self):
        self.tab_analyze = QWidget()
        layout = QVBoxLayout()
        
        # Панель управления графиками
        control_panel = QHBoxLayout()
        
        # Кнопки анализа
        self.btn_analyze = QPushButton("🔍 Автоанализ")
        self.btn_plot = QPushButton("📊 Построить график")
        
        # Выбор столбцов и типа графика
        self.cb_x_axis = QComboBox()
        self.cb_y_axis = QComboBox()
        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems([
            "Гистограмма", 
            "Boxplot", 
            "Scatter", 
            "Линейный график",
            "Круговая диаграмма"
        ])
        
        # Добавляем элементы на панель
        control_panel.addWidget(self.btn_analyze)
        control_panel.addWidget(self.btn_plot)
        control_panel.addWidget(QLabel("Тип:"))
        control_panel.addWidget(self.cb_plot_type)
        control_panel.addWidget(QLabel("X:"))
        control_panel.addWidget(self.cb_x_axis)
        control_panel.addWidget(QLabel("Y:"))
        control_panel.addWidget(self.cb_y_axis)
        
        # Графическая область
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Отчёт анализа
        self.analysis_report = QTextEdit()
        self.analysis_report.setReadOnly(True)
        
        # Сборка layout
        layout.addLayout(control_panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.analysis_report)
        self.tab_analyze.setLayout(layout)
        self.tabs.addTab(self.tab_analyze, "Анализ")
    
    def init_clean_tab(self):
        self.tab_clean = QWidget()
        self.btn_clean = QPushButton("Удалить дубликаты")
        clean_layout = QVBoxLayout()
        clean_layout.addWidget(self.btn_clean)
        clean_layout.addStretch()
        self.tab_clean.setLayout(clean_layout)
        self.tabs.addTab(self.tab_clean, "Дубликаты")
    
    def init_missing_tab(self):
        self.tab_missing = QWidget()
        fill_group = QGroupBox("Заполнение пропусков")
        fill_layout = QVBoxLayout()
        self.cb_num_strategy = QComboBox()
        self.cb_num_strategy.addItems(["mean", "median", "constant"])
        self.cb_cat_strategy = QComboBox()
        self.cb_cat_strategy.addItems(["mode", "constant"])
        self.le_fill_value = QLineEdit("NULL")
        self.le_fill_value.setPlaceholderText("Значение для 'constant'")
        self.btn_drop_missing = QPushButton("Удалить строки с пропусками")
        self.btn_process_missing = QPushButton("Обработать пропуски")
        missing_layout = QVBoxLayout()
        missing_layout.addWidget(QLabel("Стратегия для чисел:"))
        missing_layout.addWidget(self.cb_num_strategy)
        missing_layout.addWidget(QLabel("Стратегия для категорий:"))
        missing_layout.addWidget(self.cb_cat_strategy)
        missing_layout.addWidget(QLabel("Кастомное значение:"))
        missing_layout.addWidget(self.le_fill_value)
        missing_layout.addWidget(self.btn_process_missing)
        missing_layout.addStretch()
        self.tab_missing.setLayout(missing_layout)
        self.tabs.addTab(self.tab_missing, "Пропуски")
    
    def init_outliers_tab(self):
        self.tab_outliers = QWidget()
        self.le_outlier_cols = QLineEdit()
        self.le_outlier_cols.setPlaceholderText("Укажите столбцы через запятую")
        self.cb_outlier_method = QComboBox()
        self.cb_outlier_method.addItems(["IQR", "Hampel", "Percentile", "Skewness", "Kurtosis"])
        self.btn_remove_outliers = QPushButton("Удалить выбросы")
        outliers_layout = QVBoxLayout()
        outliers_layout.addWidget(QLabel("Столбцы для обработки:"))
        outliers_layout.addWidget(self.le_outlier_cols)
        outliers_layout.addWidget(QLabel("Метод обнаружения:"))
        outliers_layout.addWidget(self.cb_outlier_method)
        outliers_layout.addWidget(self.btn_remove_outliers)
        outliers_layout.addStretch()
        self.tab_outliers.setLayout(outliers_layout)
        self.tabs.addTab(self.tab_outliers, "Выбросы")
    
    def init_scaling_tab(self):
        self.tab_scaling = QWidget()
        layout = QVBoxLayout()
        self.scaling_columns = QLineEdit()
        self.scaling_columns.setPlaceholderText("Укажите столбцы через запятую")
        self.scaling_method = QButtonGroup()
        self.rb_normalize = QRadioButton("Нормализация (MinMax)")
        self.rb_standardize = QRadioButton("Стандартизация (Z-score)")
        self.rb_normalize.setChecked(True)
        self.scaling_method.addButton(self.rb_normalize)
        self.scaling_method.addButton(self.rb_standardize)
        self.norm_range_layout = QHBoxLayout()
        self.norm_range_layout.addWidget(QLabel("Диапазон:"))
        self.norm_min = QLineEdit("0")
        self.norm_max = QLineEdit("1")
        self.norm_min.setValidator(QDoubleValidator())
        self.norm_max.setValidator(QDoubleValidator())
        self.norm_range_layout.addWidget(self.norm_min)
        self.norm_range_layout.addWidget(QLabel("до"))
        self.norm_range_layout.addWidget(self.norm_max)
        self.norm_params_container = QWidget()
        self.norm_params_container.setLayout(self.norm_range_layout)
        self.btn_apply_scaling = QPushButton("Применить масштабирование")
        
        layout.addWidget(QLabel("Столбцы для обработки:"))
        layout.addWidget(self.scaling_columns)
        layout.addWidget(QLabel("Метод:"))
        layout.addWidget(self.rb_normalize)
        layout.addWidget(self.rb_standardize)
        layout.addWidget(self.norm_params_container)
        layout.addWidget(self.btn_apply_scaling)
        layout.addStretch()
        
        self.rb_normalize.toggled.connect(self.norm_params_container.setVisible)
        self.norm_params_container.setVisible(self.rb_normalize.isChecked())
        
        self.tab_scaling.setLayout(layout)
        self.tabs.addTab(self.tab_scaling, "Масштабирование")
    
    def init_text_processing_tab(self):
        """Инициализация вкладки для обработки текста с анализом"""
        self.tab_text = QWidget()
        main_layout = QVBoxLayout(self.tab_text)
        
        # Создаем главный разделитель (вертикальный)
        main_splitter = QSplitter(Qt.Vertical)
        
        # 1. Верхняя часть - текст и настройки обработки
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Область для текста с прокруткой
        text_group = QGroupBox("Работа с текстом")
        text_layout = QVBoxLayout()
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Введите текст или загрузите файл...")
        text_layout.addWidget(self.text_display)
        text_group.setLayout(text_layout)
        top_layout.addWidget(text_group)
        
        # Группа настроек обработки
        settings_group = QGroupBox("Настройки обработки")
        settings_layout = QHBoxLayout()
        
        # Колонка 1: Очистка
        clean_group = QGroupBox("Очистка")
        clean_layout = QVBoxLayout()
        self.cb_remove_html = QCheckBox("Удалить HTML-теги")
        self.cb_remove_special = QCheckBox("Удалить спецсимволы")
        self.cb_numbers_strategy = QComboBox()
        self.cb_numbers_strategy.addItems(["Оставить числа", "Удалить числа", "Заменить числа"])
        clean_layout.addWidget(self.cb_remove_html)
        clean_layout.addWidget(self.cb_remove_special)
        clean_layout.addWidget(QLabel("Обработка чисел:"))
        clean_layout.addWidget(self.cb_numbers_strategy)
        clean_group.setLayout(clean_layout)
        
        # Колонка 2: Токенизация
        token_group = QGroupBox("Токенизация")
        token_layout = QVBoxLayout()
        self.cb_tokenize = QComboBox()
        self.cb_tokenize.addItems(["Не токенизировать", "По словам", "По предложениям"])
        self.cb_language = QComboBox()
        self.cb_language.addItems(["english", "russian"])
        # Добавляем выбор типа задачи
        self.cb_task_type = QComboBox()
        self.cb_task_type.addItems([
            "Автоопределение",
            "Исправление орфографии", 
            "Анализ тональности",
            "Извлечение сущностей",
            "Суммаризация",
            "Кластеризация",
            "Машинный перевод",
            "Ответы на вопросы"
        ])
        token_layout.addWidget(QLabel("Тип токенизации:"))
        token_layout.addWidget(self.cb_tokenize)
        token_layout.addWidget(QLabel("Язык текста:"))
        token_layout.addWidget(self.cb_language)
        token_layout.addWidget(QLabel("Тип NLP задачи:"))
        token_layout.addWidget(self.cb_task_type)
        token_group.setLayout(token_layout)
        
        # Колонка 3: Нормализация
        norm_group = QGroupBox("Нормализация")
        norm_layout = QVBoxLayout()
        self.cb_remove_stopwords = QCheckBox("Удалить стоп-слова")
        self.cb_normalization = QComboBox()
        self.cb_normalization.addItems(["Не нормализовать", "Стемминг", "Лемматизация"])
        norm_layout.addWidget(self.cb_remove_stopwords)
        norm_layout.addWidget(QLabel("Метод нормализации:"))
        norm_layout.addWidget(self.cb_normalization)
        norm_group.setLayout(norm_layout)
        
        settings_layout.addWidget(clean_group)
        settings_layout.addWidget(token_group)
        settings_layout.addWidget(norm_group)
        settings_group.setLayout(settings_layout)
        top_layout.addWidget(settings_group)
        
        # Кнопки обработки
        button_layout = QHBoxLayout()
        self.btn_process_text = QPushButton("Обработать текст")
        self.btn_analyze_text = QPushButton("Анализировать текст")
        button_layout.addWidget(self.btn_process_text)
        button_layout.addWidget(self.btn_analyze_text)
        top_layout.addLayout(button_layout)
        
        # 2. Нижняя часть - результаты анализа
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        analysis_group = QGroupBox("Результаты анализа")
        analysis_layout = QVBoxLayout()
        self.text_analysis_display = QTextEdit()
        self.text_analysis_display.setReadOnly(True)
        analysis_layout.addWidget(self.text_analysis_display)
        analysis_group.setLayout(analysis_layout)
        bottom_layout.addWidget(analysis_group)
        
        # Добавляем виджеты в разделитель
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([500, 300])  # Начальные размеры областей
        
        # Добавляем разделитель в главный layout
        main_layout.addWidget(main_splitter)
        # Устанавливаем растягивание
        main_layout.setStretchFactor(main_splitter, 1)
        
        self.tabs.addTab(self.tab_text, "Работа с текстом")

    def init_result_tab(self):
        self.tab_result = QWidget()
        layout = QVBoxLayout()

        # Таблица для отображения результата (для табличных данных)
        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Текстовое поле для отображения результата (для текстовых данных)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        # Кнопка обновления
        self.btn_update_result = QPushButton("Обновить результат")
        self.btn_update_result.clicked.connect(self.update_result_display)

        layout.addWidget(QLabel("Итоговый результат:"))
        layout.addWidget(self.result_table)
        layout.addWidget(self.result_text)
        layout.addWidget(self.btn_update_result)
        self.tab_result.setLayout(layout)
        self.tabs.addTab(self.tab_result, "🏁 Результат")

    def update_result_display(self):
        """Обновляет отображение результата на вкладке"""
        self.result_table.clear()
        self.result_text.clear()

        if self.current_data is not None:
            self.result_table.setVisible(True)
            self.result_text.setVisible(False)
            self.result_table.setRowCount(self.current_data.shape[0])
            self.result_table.setColumnCount(self.current_data.shape[1])
            self.result_table.setHorizontalHeaderLabels(self.current_data.columns)
            for row in range(self.current_data.shape[0]):
                for col in range(self.current_data.shape[1]):
                    item = QTableWidgetItem(str(self.current_data.iloc[row, col]))
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    self.result_table.setItem(row, col, item)
        elif self.current_text is not None:
            self.result_table.setVisible(False)
            self.result_text.setVisible(True)
            self.result_text.setPlainText(self.current_text)
        else:
            self.result_table.setVisible(False)
            self.result_text.setVisible(True)
            self.result_text.setPlainText("Нет данных для отображения")
    
    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)
    
    def init_toolbar(self):
        toolbar = self.addToolBar("Инструменты")
        export_action = QAction("Экспорт PDF", self)
        export_action.triggered.connect(self.export_report)
        undo_action = QAction("Отменить", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(export_action)
        toolbar.addAction(undo_action)
    
    def setup_connections(self):
        self.btn_load.clicked.connect(self.load_data)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_load_text.clicked.connect(self.load_text_file)
        self.btn_clean.clicked.connect(self.clean_data)
        self.btn_process_missing.clicked.connect(self.process_missing)
        self.btn_remove_outliers.clicked.connect(self.remove_outliers)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_plot.clicked.connect(self.plot_data)
        self.btn_apply_scaling.clicked.connect(self.apply_scaling)
        self.btn_process_text.clicked.connect(self.process_text)
        self.btn_analyze_text.clicked.connect(self.analyze_text_data)
        self.cb_plot_type.currentTextChanged.connect(self.update_axis_visibility)
        self.btn_drop_missing.clicked.connect(self.drop_missing_rows)
    def drop_missing_rows(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                
                # Получаем список столбцов для обработки (можно добавить выбор столбцов)
                columns = None  # Обрабатываем все столбцы
                
                processor = HandleMissingValues(self.current_data)
                self.current_data = processor.drop_missing_rows(columns)
                
                self.display_data()
                self.save_state()
                self.log_message(f"Удалено строк с пропусками: {len(processor.data) - len(self.current_data)}")
            except Exception as e:
                self.log_message(f"Ошибка удаления строк: {str(e)}", error=True)
            finally:
                self.show_progress(False)
    def update_plot_columns(self):
        """Обновляет список доступных столбцов для графиков"""
        if self.current_data is not None:
            self.cb_x_axis.clear()
            self.cb_y_axis.clear()
            self.cb_x_axis.addItems(self.current_data.columns)
            self.cb_y_axis.addItems(self.current_data.columns)
            self.cb_y_axis.setCurrentIndex(1 if len(self.current_data.columns) > 1 else 0)
    
    def update_axis_visibility(self):
        """Скрывает/показывает выбор оси Y в зависимости от типа графика"""
        plot_type = self.cb_plot_type.currentText()
        self.cb_y_axis.setVisible(plot_type in ["Scatter", "Линейный график"])
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл данных",
            "",
            "Все поддерживаемые (*.csv *.xlsx *.json *.parquet);;"
            "CSV (*.csv);;Excel (*.xlsx);;JSON (*.json);;Parquet (*.parquet)"
        )
        if not file_path:
            return
        try:
            self.show_progress(True)
            if file_path.endswith('.csv'):
                self.current_data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.current_data = pd.read_excel(file_path)
            # elif file_path.endswith('.json'):
            #     # Пробуем разные способы чтения JSON
            #     with open(file_path, 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            #         if isinstance(data, list):
            #             self.current_data = pd.json_normalize(data)
            #         elif isinstance(data, dict):
            #             self.current_data = pd.DataFrame.from_dict(data, orient='columns')

            elif file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Случай 1: JSON — это список словарей (например, [{...}, {...}])
                        if isinstance(data, list):
                            self.current_data = pd.json_normalize(data)
                        
                        # Случай 2: JSON — это словарь с одним ключом-списком (как audio_features)
                        elif isinstance(data, dict):
                            # Ищем ключ, содержащий список словарей
                            list_keys = [k for k, v in data.items() if isinstance(v, list) and all(isinstance(i, dict) for i in v)]
                            
                            if list_keys:
                                # Берём первый подходящий ключ (например, 'audio_features')
                                self.current_data = pd.json_normalize(data[list_keys[0]])
                            else:
                                # Если нет списка словарей, пробуем развернуть как есть
                                self.current_data = pd.json_normalize(data)
                        
                        # Обновляем интерфейс
                        self.display_data()
                        self.update_plot_columns()
                        self.btn_save.setEnabled(True)
                        self.log_message(f"Данные загружены из {file_path}")
                        
                except Exception as e:
                    self.log_message(f"Ошибка загрузки JSON: {str(e)}", error=True)

            elif file_path.endswith('.parquet'):
                self.current_data = pd.read_parquet(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.display_data()
            self.update_plot_columns()
            self.btn_save.setEnabled(True)
            self.save_state()
            self.log_message(f"Данные загружены из {file_path}")
        except Exception as e:
            self.log_message(f"Ошибка загрузки: {str(e)}", error=True)
        finally:
            self.show_progress(False)
    
    def save_data(self):
        if self.current_data is None and self.current_text is None:
            return
            
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;JSON (*.json);;Parquet (*.parquet);;TXT (*.txt)"
        )
        if not file_path:
            return
            
        try:
            self.show_progress(True)
            
            if selected_filter == "CSV (*.csv)" and not file_path.endswith('.csv'):
                file_path += '.csv'
            elif selected_filter == "Excel (*.xlsx)" and not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            elif selected_filter == "JSON (*.json)" and not file_path.endswith('.json'):
                file_path += '.json'
            elif selected_filter == "Parquet (*.parquet)" and not file_path.endswith('.parquet'):
                file_path += '.parquet'
            elif selected_filter == "TXT (*.txt)" and not file_path.endswith('.txt'):
                file_path += '.txt'
                
            if file_path.endswith('.csv'):
                self.current_data.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                self.current_data.to_excel(file_path, index=False)
            elif file_path.endswith('.json'):
                self.current_data.to_json(file_path, orient='records')
            elif file_path.endswith('.parquet'):
                self.current_data.to_parquet(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_text)
            else:
                raise ValueError("Неподдерживаемый формат файла")
                
            self.log_message(f"Данные сохранены в {file_path}")
        except Exception as e:
            self.log_message(f"Ошибка сохранения: {str(e)}", error=True)
        finally:
            self.show_progress(False)
    
    def display_data(self):
        if self.current_data is not None:
            self.data_table.setRowCount(self.current_data.shape[0])
            self.data_table.setColumnCount(self.current_data.shape[1])
            self.data_table.setHorizontalHeaderLabels(self.current_data.columns)
            for row in range(self.current_data.shape[0]):
                for col in range(self.current_data.shape[1]):
                    item = QTableWidgetItem(str(self.current_data.iloc[row, col]))
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    self.data_table.setItem(row, col, item)
    
    def clean_data(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                processor = CleanData(self.current_data)
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка очистки: {str(e)}", error=True)
            finally:
                self.show_progress(False)
    
    def process_missing(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                fill_value = {col: self.le_fill_value.text() 
                             for col in self.current_data.columns}
                processor = HandleMissingValues(
                    self.current_data,
                    numeric_strategy=self.cb_num_strategy.currentText(),
                    categorical_strategy=self.cb_cat_strategy.currentText(),
                    fill_value=fill_value
                )
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка обработки пропусков: {str(e)}", error=True)
            finally:
                self.show_progress(False)
    
    def remove_outliers(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                columns = [col.strip() for col in self.le_outlier_cols.text().split(",")] if self.le_outlier_cols.text() else None
                processor = DetectAndRemoveOutliers(
                    self.current_data, 
                    columns=columns,
                    method=self.cb_outlier_method.currentText().lower()
                )
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка удаления выбросов: {str(e)}", error=True)
            finally:
                self.show_progress(False)
    
    def apply_scaling(self):
        if self.current_data is None:
            self.log_message("Нет данных для обработки", error=True)
            return
        try:
            self.show_progress(True)
            columns = [c.strip() for c in self.scaling_columns.text().split(",")] if self.scaling_columns.text() else None
            if self.rb_normalize.isChecked():
                processor = NormalizeData(
                    self.current_data,
                    columns=columns,
                    feature_range=(
                        float(self.norm_min.text()),
                        float(self.norm_max.text())
                    ))
            else:
                processor = StandardizeData(
                    self.current_data,
                    columns=columns
                )
            self.current_data = processor.run()
            self.display_data()
            self.save_state()
            self.log_message(processor.info())
        except ValueError as e:
            self.log_message(f"Ошибка ввода параметров: {str(e)}", error=True)
        except Exception as e:
            self.log_message(f"Ошибка масштабирования: {str(e)}", error=True)
        finally:
            self.show_progress(False)
    
    def run_analysis(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                temp_file = "temp_analysis.csv"
                self.current_data.to_csv(temp_file, index=False)
                detector = Detector(
                    check_abnormal=True,
                    check_missing=True,
                    check_duplicates=True,
                    check_scaling=True
                )
                outcome, abnormal, scaling = detector.check_dataframe(temp_file)
                
                # Формируем отчет
                report = "=== Анализ данных ===\n"
                report += f"Пропуски: {outcome['Missing values/Пропущенные значения']}\n"
                report += f"Дубликаты: {outcome['Duplicate values/Дубликаты значений ']}\n"
                
                # Добавляем рекомендации по методам удаления выбросов
                report += "\n--- Рекомендации по обработке выбросов ---\n"
                for col, methods in abnormal.items():
                    report += f"\nСтолбец: {col}\n"
                    
                    # Собираем информацию о методах
                    method_info = []
                    for method_name, method_data in methods.items():
                        if method_data['count'] > 0:
                            percent = method_data['count'] / len(self.current_data) * 100
                            method_info.append((
                                method_name,
                                method_data['count'],
                                percent,
                                method_data.get('threshold', ''),
                                method_data.get('direction', '')
                            ))
                    
                    # Сортируем методы по количеству найденных выбросов (по убыванию)
                    method_info.sort(key=lambda x: x[1], reverse=True)
                    
                    # Формируем рекомендации
                    if method_info:
                        best_method = method_info[0]
                        report += f"Рекомендуемый метод: {best_method[0]}\n"
                        report += f" - Найдено выбросов: {best_method[1]} ({best_method[2]:.1f}%)\n"
                        if best_method[3]:
                            report += f" - Порог: {best_method[3]}\n"
                        if best_method[4]:
                            report += f" - Направление: {best_method[4]}\n"
                        
                        # Добавляем альтернативные методы
                        if len(method_info) > 1:
                            report += "Альтернативные методы:\n"
                            for method in method_info[1:]:
                                report += f" - {method[0]}: {method[1]} выбросов ({method[2]:.1f}%)\n"
                    else:
                        report += "Выбросы не обнаружены\n"
                
                # Добавляем рекомендации по масштабированию
                report += "\n--- Рекомендации по масштабированию ---\n"
                for col, rec in scaling.items():
                    report += f"- {col}: {rec['Рекомендация']} ({', '.join(rec['причина'])})\n"
                
                self.analysis_report.setPlainText(report)
                self.plot_data()
                self.log_message("Автоанализ завершен")
            except Exception as e:
                self.log_message(f"Ошибка анализа: {str(e)}", error=True)
            finally:
                self.show_progress(False)
    
    def plot_data(self):
        if self.current_data is None or not self.cb_x_axis.currentText():
            return
        x_col = self.cb_x_axis.currentText()
        plot_type = self.cb_plot_type.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            if plot_type == "Гистограмма":
                self.current_data[x_col].plot(
                    kind='hist',
                    ax=ax,
                    bins=20,
                    edgecolor='black',
                    color='skyblue'
                )
                ax.set_ylabel("Частота")
            elif plot_type == "Boxplot":
                self.current_data[[x_col]].boxplot(
                    ax=ax,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue')
                )
                ax.set_ylabel("Значения")
            elif plot_type in ["Scatter", "Линейный график"]:
                if not self.cb_y_axis.currentText():
                    raise ValueError("Не выбрана ось Y")
                y_col = self.cb_y_axis.currentText()
                if plot_type == "Scatter":
                    self.current_data.plot.scatter(
                        x=x_col,
                        y=y_col,
                        ax=ax,
                        color='green',
                        alpha=0.6
                    )
                else:
                    self.current_data.plot.line(
                        x=x_col,
                        y=y_col,
                        ax=ax,
                        color='blue',
                        marker='o'
                    )
                ax.set_ylabel(y_col)
            elif plot_type == "Круговая диаграмма":
                if self.current_data[x_col].nunique() > 10:
                    raise ValueError("Слишком много уникальных значений для круговой диаграммы")
                self.current_data[x_col].value_counts().plot.pie(
                    ax=ax,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.Pastel1.colors
                )
                ax.set_ylabel("")
            ax.set_title(f"{plot_type}: {x_col}")
            ax.grid(True, linestyle='--', alpha=0.6)
            self.canvas.draw()
        except Exception as e:
            self.log_message(f"Ошибка построения графика: {str(e)}", error=True)
    
    def save_state(self):
        if self.current_data is not None:
            self.history.append(self.current_data.copy())
            if len(self.history) > 10:
                self.history.pop(0)
    
    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.current_data = self.history[-1].copy()
            self.display_data()
            self.log_message("Отмена последнего действия")
    
    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт отчёта", "", "PDF Files (*.pdf)")
        if path:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            doc = QTextDocument()
            doc.setPlainText(self.analysis_report.toPlainText())
            doc.print_(printer)
            self.log_message(f"Отчёт экспортирован в {path}")
    
    def show_progress(self, visible):
        self.progress_bar.setVisible(visible)
        self.progress_bar.setRange(0, 0 if visible else 1)
        QApplication.processEvents()
    
    def log_message(self, message, error=False):
        if error:
            self.log.append(f"<font color='red'>{message}</font>")
        else:
            self.log.append(message)
    
    def auto_detect_language(self):
        """Автоматическое определение языка текста"""
        if not self.current_text:
            return
        try:
            # Используем TextDetector для определения языка
            detector = TextDetector()
            detected_lang = detector._detect_language(self.current_text)
            # Устанавливаем соответствующий язык в выпадающем списке
            lang_mapping = {
                'english': 'english',
                'russian': 'russian',
            }
            # Если определенный язык есть в нашем списке, выбираем его
            if detected_lang in lang_mapping.values():
                index = self.cb_language.findText(detected_lang)
                if index >= 0:
                    self.cb_language.setCurrentIndex(index)
                    self.log_message(f"Автоматически определен язык: {detected_lang}")
            else:
                self.log_message(f"Определен язык: {detected_lang}, но он не поддерживается")
        except Exception as e:
            self.log_message(f"Ошибка определения языка: {str(e)}")
    
    def load_text_file(self):
        """Загрузка текстового файла с автоматическим определением языка"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите текстовый файл", 
            "", 
            "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                # Очищаем текущий текст
                self.current_text = None
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.current_text = f.read()
                elif file_path.endswith('.csv'):
                    # Читаем CSV, преобразуем в удобный текстовый формат
                    df = pd.read_csv(file_path)
                    # Формируем текст из названий столбцов и данных
                    columns = " | ".join(df.columns)
                    data_rows = []
                    for _, row in df.iterrows():
                        row_str = " | ".join(str(value) for value in row.values)
                        data_rows.append(row_str)
                    self.current_text = f"Columns: {columns}\n" + "\n".join(data_rows)
                elif file_path.endswith('.xlsx'):
                    # Читаем Excel, преобразуем в удобный текстовый формат
                    df = pd.read_excel(file_path)
                    columns = " | ".join(df.columns)
                    data_rows = []
                    for _, row in df.iterrows():
                        row_str = " | ".join(str(value) for value in row.values)
                        data_rows.append(row_str)
                    self.current_text = f"Columns: {columns}\n" + "\n".join(data_rows)
                elif file_path.endswith('.json'):
                    # Читаем JSON, преобразуем в текст
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self.current_text = "\n".join(f"{k}: {v}" for k, v in data.items())
                        elif isinstance(data, list):
                            self.current_text = "\n".join(str(item) for item in data)
                        else:
                            self.current_text = str(data)
                if self.current_text is not None:
                    self.text_display.setPlainText(self.current_text)
                    self.btn_save.setEnabled(True)
                    self.log_message(f"Текст загружен из {file_path}")
                    # Автоматическое определение языка
                    self.auto_detect_language()
            except Exception as e:
                self.log_message(f"Ошибка загрузки файла: {str(e)}", error=True)
    
    def analyze_text_data(self):
        """Анализ текста и вывод рекомендаций с учетом типа задачи"""
        if not self.current_text:
            self.log_message("Нет текста для анализа", error=True)
            return
        try:
            self.show_progress(True)
            # Получаем выбранные настройки из интерфейса
            selected_lang = self.cb_language.currentText()
            # Маппинг выбранного типа задачи на значения TextDetector
            task_type_mapping = {
                "Автоопределение": "unknown",
                "Исправление орфографии": "spelling",
                "Анализ тональности": "sentiment",
                "Извлечение сущностей": "ner",
                "Суммаризация": "summarization",
                "Кластеризация": "clustering",
                "Машинный перевод": "translation",
                "Ответы на вопросы": "question_answering"
            }
            selected_task_type = task_type_mapping.get(self.cb_task_type.currentText(), "unknown")
            # Создаем анализатор текста с учетом типа задачи
            analyzer = TextDetector(lang=selected_lang, task_type=selected_task_type)
            # Анализируем текст
            analysis_result = analyzer.analyze_text(str(self.current_text))
            # Формируем отчет
            report = "=== Анализ текста ===\n"
            report += f"📏 Длина текста: {len(self.current_text)} символов\n"
            report += f"🌍 Определенный язык: {analysis_result['detected_lang']}\n"
            report += f"⚙️ Выбранный язык: {selected_lang}\n"
            report += f"🎯 Тип задачи: {self.cb_task_type.currentText()}\n"
            if 'stopword_ratio' in analysis_result and analysis_result['stopword_ratio'] is not None:
                ratio = analysis_result['stopword_ratio']
                report += f"🛑 Доля стоп-слов: {ratio:.1%} ({ratio:.4f})\n"
            if 'stem_vs_lemma' in analysis_result and analysis_result['stem_vs_lemma'] is not None:
                diff = analysis_result['stem_vs_lemma']['stem_diff']
                total = analysis_result['stem_vs_lemma']['total_words']
                report += f"🔤 Различия стемминга/лемматизации: {diff} из {total} слов ({diff/total:.1%})\n"
            report += "\n💡 Рекомендации:\n"
            for i, rec in enumerate(analysis_result.get('recommendations', []), 1):
                # Используем action_ru вместо action
                report += f"{i}. {rec['description']}\n   → Действие: {rec.get('action_ru', rec['action'])}\n"
            # Добавляем статистику
            if 'word_stats' in analysis_result:
                stats = analysis_result['word_stats']
                report += "\n📊 Статистика:\n"
                report += f" - Уникальных слов: {stats['unique_words']}\n"
                report += f" - Средняя длина слова: {stats['avg_word_length']:.2f} символов\n"
                report += f" - Частые слова: {', '.join(stats['frequent_words'][:5])}\n"
            # Выводим отчет в новое окно анализа
            self.text_analysis_display.setPlainText(report)
            self.log_message("Анализ текста завершен")
        except Exception as e:
            self.log_message(f"Ошибка анализа текста: {str(e)}", error=True)
        finally:
            self.show_progress(False)
    
    def process_text(self):
        """Обработка текста с учетом типа задачи"""
        if not self.current_text:
            self.log_message("Нет текста для обработки", error=True)
            return
        try:
            self.show_progress(True)
            # Сохраняем исходный текст для истории
            original_text = self.current_text
            
            # Получаем выбранные настройки из интерфейса
            selected_lang = self.cb_language.currentText()
            
            # Маппинг типа задачи
            task_type_mapping = {
                "Автоопределение": "unknown",
                "Исправление орфографии": "spelling",
                "Анализ тональности": "sentiment",
                "Извлечение сущностей": "ner",
                "Суммаризация": "summarization",
                "Кластеризация": "clustering",
                "Машинный перевод": "translation",
                "Ответы на вопросы": "question_answering"
            }
            selected_task_type = task_type_mapping.get(self.cb_task_type.currentText(), "unknown")
            
            # Создаем процессор и добавляем выбранные операции
            processor = TextProcessor()
            
            # Добавляем операции
            if self.cb_remove_html.isChecked():
                processor.add_operation('remove_html')
            if self.cb_remove_special.isChecked():
                processor.add_operation('remove_special_chars')
            if self.cb_numbers_strategy.currentText() != "Оставить числа":
                strategy = 'remove' if self.cb_numbers_strategy.currentText() == "Удалить числа" else 'replace'
                processor.add_operation('handle_numbers', strategy=strategy)
            if self.cb_tokenize.currentText() != "Не токенизировать":
                token_type = self.cb_tokenize.currentText()  # "По словам" или "По предложениям"
                processor.add_operation('tokenize', 
                                    token_type=token_type,
                                    lang=selected_lang)
            if self.cb_remove_stopwords.isChecked():
                processor.add_operation('remove_stopwords',
                                    lang=selected_lang)
            if self.cb_normalization.currentText() != "Не нормализовать":
                method = 'stem' if self.cb_normalization.currentText() == "Стемминг" else 'lemmatize'
                processor.add_operation('normalize',
                                    method=method,
                                    lang=selected_lang)
            
            # Выполняем все операции
            processed_result = processor.execute(str(self.current_text))
            
            # Обновляем текущий текст
            self.current_text = processed_result
            
            # Отображаем результат
            if isinstance(processed_result, (list, pd.Series)):
                # Для токенизированного текста делаем красивое отображение
                display_text = "\n".join([f"- {token}" for token in processed_result]) if isinstance(processed_result, list) else processed_result.to_string()
                self.text_display.setPlainText(display_text)
            else:
                self.text_display.setPlainText(str(self.current_text))
                
            self.log_message("Текст успешно обработан")
            
            # Сохраняем в историю
            self.history.append({
                'original': original_text,
                'processed': self.current_text
            })
        except Exception as e:
            self.log_message(f"Ошибка обработки: {str(e)}", error=True)
        finally:
            self.show_progress(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Настройка стиля
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())
    