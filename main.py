import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QInputDialog, QMessageBox
from PyQt5.QtGui import QGuiApplication, QIcon

from main_ui import Ui_MainWindow

QGuiApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QGuiApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class MainWidget(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("NumHeaven - Генератор синтетических числовых данных")
        icon_path = "images/app_icon.jpg"
        self.setWindowIcon(QIcon(icon_path))

        self.generated_data = np.array([])
        self.column_names = []
        self.column_settings = []
        self.is_dirty = False

        self._set_initial_ui_values()
        self.setup_connections()
        self.toggle_precision()
        self._add_tooltips()

    def _set_initial_ui_values(self):
        """Устанавливает начальные значения для элементов UI, при первом запуске или после полной очистки."""
        self.quantitySpinBox.setValue(10)
        self.columnsSpinBox.setValue(5)
        self.minSpinBox.setValue(0)
        self.maxSpinBox.setValue(10)
        self.dataTypeComboBox.setCurrentIndex(0)
        self.precisionSpinBox.setValue(1)
        self.distributionComboBox.setCurrentIndex(0)

        self.addOutliersCheckBox.setChecked(False)
        self.outlierPercentageSpinBox.setValue(5.0)
        self.outlierMagnitudeSpinBox.setValue(3.0)
        self.graphBinsSpinBox.setValue(30)
        self.searchLineEdit.clear()

    def _add_tooltips(self):
        """Добавляет всплывающие подсказки к элементам UI."""
        self.generateButton.setToolTip(
            "Сгенерировать данные, используя текущие индивидуальные настройки столбцов.\n"
            "Если количество строк/столбцов изменено, адаптирует их."
        )
        self.regenerateButton.setToolTip(
            "Перегенерировать всю таблицу с нуля, используя глобальные настройки из UI для всех столбцов.\n"
            "Индивидуальные настройки столбцов будут сброшены."
        )
        self.applyButton.setToolTip(
            "Применить текущие глобальные настройки из UI только к выделенным столбцам таблицы."
        )
        self.quantitySpinBox.setToolTip(
            "Количество генерируемых строк."
        )
        self.columnsSpinBox.setToolTip(
            "Количество генерируемых столбцов."
        )
        self.minSpinBox.setToolTip(
            "Минимальное значение для генерации."
        )
        self.maxSpinBox.setToolTip(
            "Максимальное значение для генерации."
        )
        self.dataTypeComboBox.setToolTip(
            "Тип генерируемых данных."
        )
        self.precisionSpinBox.setToolTip(
            "Количество знаков после запятой для float."
        )
        self.distributionComboBox.setToolTip(
            "Тип распределения для генерируемых данных."
        )
        self.saveButton.setToolTip(
            "Сохранить текущую таблицу данных в файл (CSV или Excel)."
        )
        self.loadButton.setToolTip(
            "Загрузить таблицу данных из файла (CSV или Excel)."
        )
        self.saveConfigButton.setToolTip(
            "Сохранить текущую конфигурацию генерации (настройки столбцов, имена и т.д.)."
        )
        self.loadConfigButton.setToolTip(
            "Загрузить ранее сохраненную конфигурацию генерации."
        )
        self.showButton.setToolTip(
            "Показать гистограмму распределения для выбранных столбцов (или для всех данных)."
        )
        self.addOutliersCheckBox.setToolTip(
            "Включить добавление выбросов в сгенерированные данные."
        )
        self.outlierPercentageSpinBox.setToolTip(
            "Процент строк, которые будут заменены выбросами."
        )
        self.outlierMagnitudeSpinBox.setToolTip(
            "Множитель стандартного отклонения для определения величины выброса."
        )
        self.clearAllButton.setToolTip(
            "Очистить таблицу, сбросить все имена и настройки столбцов."
        )
        self.graphBinsSpinBox.setToolTip(
            "Количество столбцов (бинов) для построения гистограммы."
        )

    def setup_connections(self):
        """Настраивает соединения элементов."""
        self.generateButton.clicked.connect(self.existing_settings_generation)
        self.regenerateButton.clicked.connect(self.global_settings_generation)
        self.applyButton.clicked.connect(self.apply_settings)

        self.saveButton.clicked.connect(self.save_table_to_file)
        self.loadButton.clicked.connect(self.load_table_from_file)
        self.saveConfigButton.clicked.connect(self.save_configuration)
        self.loadConfigButton.clicked.connect(self.load_configuration)

        self.showButton.clicked.connect(self.show_graph)
        self.dataTypeComboBox.currentTextChanged.connect(self.toggle_precision)
        self.dataTable.itemSelectionChanged.connect(self.update_ui_from_selected_column)
        self.dataTable.horizontalHeader().sectionDoubleClicked.connect(self.rename_column_header)
        self.searchPushButton.clicked.connect(self.search_column)
        self.clearAllButton.clicked.connect(self.clear_all_data_and_settings)

        spin_boxes_to_track = [
            self.quantitySpinBox, self.columnsSpinBox, self.minSpinBox,
            self.maxSpinBox, self.precisionSpinBox,
            self.outlierPercentageSpinBox, self.outlierMagnitudeSpinBox
        ]
        for spin_box in spin_boxes_to_track:
            spin_box.valueChanged.connect(self._mark_dirty)

        combo_boxes_to_track = [
            self.dataTypeComboBox, self.distributionComboBox
        ]
        for combo_box in combo_boxes_to_track:
            combo_box.currentIndexChanged.connect(self._mark_dirty)

        self.addOutliersCheckBox.stateChanged.connect(self._mark_dirty)

    def _mark_dirty(self):
        """Устанавливает флаг, что конфигурация была изменена."""
        self.is_dirty = True

    def _mark_clean(self):
        """Сбрасывает флаг измененной конфигурации."""
        self.is_dirty = False

    def toggle_precision(self):
        """Включает/выключает QSpinBox для точности."""
        is_integer = self.dataTypeComboBox.currentText().lower() == "integer"
        self.precisionSpinBox.setDisabled(is_integer)
        if is_integer:
            style_sheet = "border: 1px solid gray; background-color: #f0f0f0;"
        else:
            style_sheet = ""
        self.precisionSpinBox.setStyleSheet(style_sheet)

    def _get_global_ui_settings(self):
        """Собирает текущие глобальные настройки из элементов UI."""
        precision_value = 0
        if self.precisionSpinBox.isEnabled():
            precision_value = self.precisionSpinBox.value()

        settings = {
            "min": self.minSpinBox.value(),
            "max": self.maxSpinBox.value(),
            "type": self.dataTypeComboBox.currentText(),
            "precision": precision_value,
            "distribution": self.distributionComboBox.currentText(),
            "add_outliers": self.addOutliersCheckBox.isChecked(),
            "outlier_percentage": self.outlierPercentageSpinBox.value(),
            "outlier_magnitude": self.outlierMagnitudeSpinBox.value()
        }

        if settings["distribution"].lower() == "beta":
            settings["beta_a"] = 2.0
            settings["beta_b"] = 2.0

        return settings

    def _validate_settings(self, settings, column_name=""):
        """Проверяет корректность настроек."""
        if column_name:
            prefix = f"Для столбца '{column_name}': "
        else:
            prefix = "В глобальных настройках: "

        if settings["min"] >= settings["max"]:
            QMessageBox.warning(
                self,
                "Ошибка в настройках",
                f"{prefix}Минимальное значение ({settings['min']}) "
                f"должно быть меньше максимального ({settings['max']})."
            )
            return False

        add_outliers_setting = settings.get("add_outliers", False)
        if add_outliers_setting:
            outlier_percentage = settings.get("outlier_percentage", 0)
            if not (0 <= outlier_percentage <= 100):
                QMessageBox.warning(
                    self,
                    "Ошибка в настройках выбросов",
                    f"{prefix}Процент выбросов должен быть от 0 до 100."
                )
                return False

            outlier_magnitude = settings.get("outlier_magnitude", 0.1)
            if outlier_magnitude < 0.1:
                QMessageBox.warning(
                    self,
                    "Ошибка в настройках выбросов",
                    f"{prefix}Множитель силы выбросов должен быть не меньше 0.1."
                )
                return False

        if settings["distribution"] == "lognormal" and settings["min"] < 0:
            if column_name:
                prefix_lognormal = f"Для столбца '{column_name}': "
            else:
                prefix_lognormal = "В глобальных настройках: "

            QMessageBox.warning(
                self,
                "Ошибка в настройках",
                f"{prefix_lognormal}Для логнормального распределения минимальное значение "
                f"({settings['min']}) не может быть отрицательным."
            )
            return False

        return True

    def _apply_outliers(self, column_data_array, column_settings):
        """Применяет выбросы к данным столбца."""
        add_outliers_flag = column_settings.get("add_outliers", False)
        outlier_percentage = column_settings.get("outlier_percentage", 0)

        if not add_outliers_flag or outlier_percentage == 0:
            return column_data_array

        num_samples = len(column_data_array)
        num_outliers = int(round(num_samples * (outlier_percentage / 100.0)))

        if num_outliers == 0 or num_samples == 0:
            return column_data_array

        num_outliers = min(num_outliers, num_samples)

        outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)

        valid_data_for_stats = column_data_array[~pd.isna(column_data_array)]
        if len(valid_data_for_stats) == 0:
            return column_data_array

        data_mean = np.mean(valid_data_for_stats)
        data_std = np.std(valid_data_for_stats)
        if data_std == 0:
            data_std = 1.0

        outlier_strength = column_settings.get("outlier_magnitude", 3.0)

        for idx in outlier_indices:
            direction = np.random.choice([-1, 1])
            outlier_value = data_mean + direction * outlier_strength * data_std

            if column_settings["type"].lower() == "integer":
                column_data_array[idx] = int(round(outlier_value))
            elif column_settings["type"].lower() == "float":
                column_data_array[idx] = round(outlier_value, column_settings["precision"])

        return column_data_array

    def _generate_single_column(self, settings, quantity):
        """Генерирует данные для одного столбца."""
        min_val = settings["min"]
        max_val = settings["max"]
        dtype = settings["type"].lower()
        precision = settings["precision"]
        distribution = settings["distribution"].lower()

        column_array = None

        if quantity <= 0:
            return np.array([])

        std_dev_val = (max_val - min_val) / 6
        scale_val = (max_val - min_val) / 3

        if dtype == "integer":
            if distribution == "uniform":
                column_array = np.random.randint(min_val, max_val + 1, quantity)
            elif distribution == "normal":
                mean_val = (min_val + max_val) / 2
                generated_data = np.random.normal(mean_val, std_dev_val, quantity)
                column_array = np.round(generated_data).astype(int)
                column_array = np.clip(column_array, min_val, max_val)
            elif distribution == "exponential":
                generated_data = np.random.exponential(scale_val, quantity) + min_val
                column_array = np.round(generated_data).astype(int)
                column_array = np.clip(column_array, min_val, max_val)
            elif distribution == "lognormal":
                eff_min = max(1e-9, min_val)
                eff_max = max(eff_min + 1.0, max_val)
                log_min = np.log(eff_min)
                log_max = np.log(eff_max)
                mu = (log_min + log_max) / 2
                sigma = (log_max - log_min) / 6
                if sigma <= 0:
                    sigma = 0.1
                generated_data = np.random.lognormal(mu, sigma, quantity)
                column_array = np.round(generated_data).astype(int)
                column_array = np.clip(column_array, min_val, max_val)
            elif distribution == "beta":
                beta_a = settings.get("beta_a", 2.0)
                beta_b = settings.get("beta_b", 2.0)
                generated_data = np.random.beta(beta_a, beta_b, quantity)
                column_array = min_val + generated_data * (max_val - min_val)
                column_array = np.round(column_array).astype(int)
                column_array = np.clip(column_array, min_val, max_val)

        elif dtype == "float":
            temp_data = None
            if distribution == "uniform":
                temp_data = np.random.uniform(min_val, max_val, quantity)
            elif distribution == "normal":
                mean_val = (min_val + max_val) / 2
                temp_data = np.random.normal(mean_val, std_dev_val, quantity)
            elif distribution == "exponential":
                temp_data = np.random.exponential(scale_val, quantity) + min_val
            elif distribution == "lognormal":
                eff_min = max(1e-9, min_val)
                eff_max = max(eff_min + 1e-6, max_val)
                log_min = np.log(eff_min)
                log_max = np.log(eff_max)
                mu = (log_min + log_max) / 2
                sigma = (log_max - log_min) / 6
                if sigma <= 0:
                    sigma = 0.1
                temp_data = np.random.lognormal(mu, sigma, quantity)
            elif distribution == "beta":
                beta_a = settings.get("beta_a", 2.0)
                beta_b = settings.get("beta_b", 2.0)
                beta_values = np.random.beta(beta_a, beta_b, quantity)
                temp_data = min_val + beta_values * (max_val - min_val)

            if temp_data is not None:
                column_array = np.round(temp_data, precision)
                column_array = np.clip(column_array, min_val, max_val)

        if column_array is None:
            QMessageBox.warning(
                self,
                "Ошибка генерации",
                f"Неизвестное распределение: {distribution} для типа {dtype}."
            )
            return np.full(quantity, np.nan)

        column_array = self._apply_outliers(column_array, settings)
        return column_array

    def global_settings_generation(self):
        """Кнопка 'Regenerate': Пересоздает таблицу с глобальными настройками UI."""
        global_ui_settings = self._get_global_ui_settings()
        if not self._validate_settings(global_ui_settings, "глобальных настроек UI"):
            return

        has_custom_settings = False
        if self.column_settings:
            for s_col in self.column_settings:
                keys_to_compare = [
                    "min", "max", "type", "precision", "distribution",
                    "add_outliers", "outlier_percentage", "outlier_magnitude",
                    "beta_a", "beta_b"
                ]
                for key in keys_to_compare:
                    if s_col.get(key) != global_ui_settings.get(key):
                        has_custom_settings = True
                        break
                if has_custom_settings:
                    break

        if has_custom_settings:
            msg_box_confirm = QMessageBox(self)
            msg_box_confirm.setWindowTitle('Подтверждение')
            msg_box_confirm.setText("Обнаружены индивидуальные настройки столбцов. "
                                    "Они будут сброшены. Продолжить?")
            msg_box_confirm.setIcon(QMessageBox.Question)

            yes_btn = msg_box_confirm.addButton("да", QMessageBox.YesRole)
            no_btn = msg_box_confirm.addButton("нет", QMessageBox.NoRole)
            msg_box_confirm.setDefaultButton(no_btn)
            msg_box_confirm.setEscapeButton(no_btn)

            msg_box_confirm.exec_()

            if msg_box_confirm.clickedButton() != yes_btn:
                return

        num_columns = self.columnsSpinBox.value()
        quantity = self.quantitySpinBox.value()

        if quantity <= 0 or num_columns <= 0:
            QMessageBox.warning(self, "Ошибка", "Количество строк и столбцов должно быть > 0.")
            return

        self.column_names = [f"Столбец {i + 1}" for i in range(num_columns)]
        self.column_settings = [global_ui_settings.copy() for _ in range(num_columns)]

        all_generated_data = []
        for i in range(num_columns):
            settings_for_col = self.column_settings[i]
            col_data = self._generate_single_column(settings_for_col, quantity)
            all_generated_data.append(col_data)

        if not all_generated_data:
            default_shape_rows = quantity if quantity > 0 else 0
            self.generated_data = np.array([]).reshape(default_shape_rows, 0)
        elif any(isinstance(d, np.ndarray) and d.size == 0 and quantity > 0 for d in all_generated_data):
            default_shape_rows = quantity if quantity > 0 else 0
            self.generated_data = np.array([]).reshape(default_shape_rows, 0)
        else:
            self.generated_data = np.column_stack(all_generated_data)

        self.update_table_widget()
        self.statusbar.showMessage("Таблица перегенерирована с глобальными настройками.", 3000)
        self._mark_dirty()

    def existing_settings_generation(self):
        """Кнопка 'Generate': Генерирует/обновляет данные с текущими настройками столбцов."""
        new_quantity = self.quantitySpinBox.value()
        new_num_columns = self.columnsSpinBox.value()

        if new_quantity <= 0 or new_num_columns <= 0:
            QMessageBox.warning(self, "Ошибка", "Количество строк и столбцов должно быть > 0.")
            return

        global_ui_settings_new = self._get_global_ui_settings()

        current_num_names = len(self.column_names)
        if new_num_columns > current_num_names:
            for i in range(current_num_names, new_num_columns):
                self.column_names.append(f"Столбец {i + 1}")
        elif new_num_columns < current_num_names:
            self.column_names = self.column_names[:new_num_columns]

        current_num_settings = len(self.column_settings)
        if new_num_columns > current_num_settings:
            if not self._validate_settings(global_ui_settings_new, "для новых столбцов"):
                return
            for _ in range(current_num_settings, new_num_columns):
                self.column_settings.append(global_ui_settings_new.copy())
        elif new_num_columns < current_num_settings:
            self.column_settings = self.column_settings[:new_num_columns]

        if not self.column_settings and new_num_columns > 0:
            if not self._validate_settings(global_ui_settings_new, "начальных настроек"):
                return
            self.column_settings = [global_ui_settings_new.copy() for _ in range(new_num_columns)]
            if not self.column_names:
                self.column_names = [f"Столбец {i + 1}" for i in range(new_num_columns)]

        all_generated_data = []
        is_generation_valid = True
        for i in range(new_num_columns):
            col_name_for_message = self.column_names[i] if i < len(self.column_names) else f"Столбец {i + 1}"
            if not self._validate_settings(self.column_settings[i], col_name_for_message):
                is_generation_valid = False
                all_generated_data.append(np.full(new_quantity, np.nan))
                continue

            col_data = self._generate_single_column(self.column_settings[i], new_quantity)
            all_generated_data.append(col_data)

        if not all_generated_data:
            default_shape_rows = new_quantity if new_quantity > 0 else 0
            self.generated_data = np.array([]).reshape(default_shape_rows, 0)
        elif any(isinstance(d, np.ndarray) and d.size == 0 and new_quantity > 0 for d in all_generated_data):
            default_shape_rows = new_quantity if new_quantity > 0 else 0
            self.generated_data = np.array([]).reshape(default_shape_rows, 0)
        else:
            self.generated_data = np.column_stack(all_generated_data)

        status_message = "Данные обновлены."
        if not is_generation_valid:
            status_message = "Генерация с ошибками в настройках некоторых столбцов."
        self.statusbar.showMessage(status_message, 3000)
        self.update_table_widget()
        self._mark_dirty()

    def apply_settings(self):
        """Применяет глобальные настройки UI к выделенным столбцам."""
        selected_indices = self.get_selected_indices()
        if not selected_indices:
            QMessageBox.information(self, "Нет выбора", "Выберите столбцы.")
            return

        if self.generated_data.size == 0:
            QMessageBox.warning(self, "Нет данных", "Сгенерируйте таблицу.")
            return

        ui_settings = self._get_global_ui_settings()
        if not self._validate_settings(ui_settings, "из UI для применения"):
            return

        quantity = self.generated_data.shape[0]
        for col_idx in selected_indices:
            if 0 <= col_idx < len(self.column_settings) and \
                    0 <= col_idx < self.generated_data.shape[1]:
                self.column_settings[col_idx] = ui_settings.copy()
                new_column_data = self._generate_single_column(ui_settings, quantity)
                self.generated_data[:, col_idx] = new_column_data

                col_name = self.column_names[col_idx] if col_idx < len(self.column_names) else f"Столбец {col_idx + 1}"
                self.statusbar.showMessage(f"Настройки применены к '{col_name}'.", 2000)

        self.update_table_widget()
        self._mark_dirty()

    def update_ui_from_selected_column(self):
        """Загружает настройки выделенного столбца в UI."""
        selected_indices = self.get_selected_indices()
        if not selected_indices:
            return

        first_idx = selected_indices[0]

        if 0 <= first_idx < len(self.column_settings):
            settings = self.column_settings[first_idx]

            self.minSpinBox.setValue(settings.get("min", 0))
            self.maxSpinBox.setValue(settings.get("max", 100))
            self.dataTypeComboBox.setCurrentText(settings.get("type", "float"))
            self.toggle_precision()
            if settings.get("type", "float").lower() == "float":
                self.precisionSpinBox.setValue(settings.get("precision", 1))
            self.distributionComboBox.setCurrentText(settings.get("distribution", "uniform"))

            self.addOutliersCheckBox.setChecked(settings.get("add_outliers", False))
            self.outlierPercentageSpinBox.setValue(settings.get("outlier_percentage", 5.0))
            self.outlierMagnitudeSpinBox.setValue(settings.get("outlier_magnitude", 3.0))

            col_name = self.column_names[first_idx] if first_idx < len(
                self.column_names) else f"Столбец {first_idx + 1}"
            info_parts = [
                f"Столбец: '{col_name}'",
                f"Тип: {settings.get('type')}",
                f"Распред: {settings.get('distribution')}",
                f"Min: {settings.get('min')}, Max: {settings.get('max')}"
            ]
            if settings.get('type', 'float').lower() == 'float':
                info_parts.append(f"Точность: {settings.get('precision')}")
            if settings.get("add_outliers"):
                info_parts.append(
                    f"Выбросы: {settings.get('outlier_percentage')}% ({settings.get('outlier_magnitude')}x)")

            self.statusbar.showMessage(" | ".join(info_parts), 7000)

    def get_selected_indices(self):
        """Возвращает список индексов выделенных столбцов."""
        selected_model_indices = self.dataTable.selectedIndexes()
        if not selected_model_indices:
            return []
        return sorted(list(set(idx.column() for idx in selected_model_indices)))

    def rename_column_header(self, logical_index):
        """Переименовывает столбец."""
        if not (0 <= logical_index < len(self.column_names)):
            return

        old_name = self.column_names[logical_index]
        new_name, ok = QInputDialog.getText(
            self,
            "Переименовать столбец",
            f"Новое имя для '{old_name}':",
            QtWidgets.QLineEdit.Normal,
            old_name
        )
        if ok:
            new_name_stripped = new_name.strip()
            if not new_name_stripped:
                QMessageBox.warning(self, "Ошибка ввода", "Имя столбца не может быть пустым.")
                return

            self.column_names[logical_index] = new_name_stripped
            self.update_table_widget()
            self.statusbar.showMessage(f"Столбец '{old_name}' переименован в '{new_name_stripped}'.", 3000)
            self._mark_dirty()

    def search_column(self):
        """Ищет и выделяет столбец."""
        search_text = self.searchLineEdit.text().strip().lower()
        if not search_text:
            self.dataTable.clearSelection()
            return

        for i, name in enumerate(self.column_names):
            if search_text in name.lower():
                self.dataTable.selectColumn(i)
                self.statusbar.showMessage(f"Найден столбец: {name}", 2000)
                return

        self.dataTable.clearSelection()
        self.statusbar.showMessage(f"Столбец по запросу '{search_text}' не найден.", 2000)

    def save_table_to_file(self):
        """Сохраняет таблицу в CSV или Excel."""
        if self.generated_data.size == 0:
            QMessageBox.information(self, "Нет данных", "Таблица пуста. Нечего сохранять.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить таблицу",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not file_path:
            return

        df = pd.DataFrame(self.generated_data, columns=self.column_names)
        try:
            for i, name in enumerate(self.column_names):
                if self.column_settings and i < len(self.column_settings) and \
                        self.column_settings[i]["type"].lower() == "integer":
                    if not df[name].isnull().any():
                        df[name] = df[name].astype(int)

            if selected_filter == "Excel Files (*.xlsx)":
                if not file_path.lower().endswith(".xlsx"):
                    file_path += ".xlsx"
                df.to_excel(file_path, index=False, engine='openpyxl')
                save_message = f"Таблица сохранена в Excel: {file_path}"
            else:
                if not file_path.lower().endswith(".csv"):
                    file_path += ".csv"
                df.to_csv(file_path, index=False)
                save_message = f"Таблица сохранена в CSV: {file_path}"

            self.statusbar.showMessage(save_message, 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл: {e}")

    def load_table_from_file(self):
        """Загружает таблицу из CSV или Excel."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить таблицу",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                QMessageBox.warning(self, "Ошибка формата", "Неверный формат файла. Поддерживаются CSV и Excel.")
                return

            self.generated_data = df.to_numpy(na_value=np.nan)
            self.column_names = df.columns.tolist()

            num_cols = len(self.column_names)
            num_rows = len(df)

            if num_cols == 0 or num_rows == 0:
                QMessageBox.information(self, "Пустой файл", "Загруженный файл не содержит данных.")
                self.clear_all_data_and_settings()
                return

            self.columnsSpinBox.setValue(num_cols)
            self.quantitySpinBox.setValue(num_rows)

            base_settings_template = self._get_global_ui_settings()
            new_settings_list = []
            for col_name_iter in self.column_names:
                col_series = df[col_name_iter]
                current_col_settings = base_settings_template.copy()
                current_col_settings["add_outliers"] = False

                if col_series.isnull().all():
                    current_col_settings["type"] = "float"
                    current_col_settings["min"], current_col_settings["max"] = 0, 1
                elif pd.api.types.is_integer_dtype(col_series.infer_objects()) and not col_series.isnull().any():
                    current_col_settings["type"] = "integer"
                    current_col_settings["precision"] = 0
                elif pd.api.types.is_float_dtype(col_series.infer_objects()) or \
                        (pd.api.types.is_numeric_dtype(col_series.infer_objects()) and col_series.isnull().any()):
                    current_col_settings["type"] = "float"

                if pd.api.types.is_numeric_dtype(col_series.infer_objects()) and not col_series.isnull().all():
                    actual_min_val = col_series.min(skipna=True)
                    actual_max_val = col_series.max(skipna=True)
                    current_col_settings["min"] = actual_min_val
                    current_col_settings["max"] = actual_max_val
                    if current_col_settings["type"] == "integer":
                        current_col_settings["min"] = int(round(actual_min_val))
                        current_col_settings["max"] = int(round(actual_max_val))

                    if current_col_settings["min"] >= current_col_settings["max"]:
                        increment = 1 if current_col_settings["type"] == "integer" else 0.1
                        current_col_settings["max"] = current_col_settings["min"] + increment
                elif not pd.api.types.is_numeric_dtype(col_series.infer_objects()):
                    current_col_settings["min"], current_col_settings["max"] = 0, 1

                new_settings_list.append(current_col_settings)

            self.column_settings = new_settings_list
            self.update_table_widget()
            self.statusbar.showMessage(f"Таблица загружена из {file_path}", 3000)
            self._mark_dirty()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить файл: {e}")

    def save_configuration(self):
        """Сохраняет конфигурацию в JSON."""
        if not self.column_settings and not self.column_names:
            QMessageBox.information(
                self,
                "Нет конфигурации",
                "Нет данных для сохранения конфигурации. Сначала сгенерируйте данные."
            )
            return

        config_data = {
            "quantity": self.quantitySpinBox.value(),
            "num_columns": self.columnsSpinBox.value(),
            "column_names": self.column_names,
            "column_settings": self.column_settings,
            "global_ui_settings": self._get_global_ui_settings()
        }

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить конфигурацию",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                if not file_path.lower().endswith(".json"):
                    file_path += ".json"
                with open(file_path, "w", encoding='utf-8') as f_out:
                    json.dump(config_data, f_out, indent=4, ensure_ascii=False)
                self.statusbar.showMessage(f"Конфигурация сохранена: {file_path}", 3000)
                self._mark_clean()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить конфигурацию: {e}")

    def load_configuration(self):
        """Загружает конфигурацию из JSON."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить конфигурацию",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r", encoding='utf-8') as f_in:
                    config_data = json.load(f_in)

                self.quantitySpinBox.setValue(config_data.get("quantity", 10))
                num_cols_from_config = config_data.get("num_columns", 1)
                self.columnsSpinBox.setValue(num_cols_from_config)

                self.column_names = config_data.get("column_names", [])
                self.column_settings = config_data.get("column_settings", [])

                global_ui_settings_from_config = config_data.get("global_ui_settings")
                if global_ui_settings_from_config:
                    self.minSpinBox.setValue(global_ui_settings_from_config.get("min", 0))
                    self.maxSpinBox.setValue(global_ui_settings_from_config.get("max", 10))
                    self.dataTypeComboBox.setCurrentText(global_ui_settings_from_config.get("type", "float"))
                    self.toggle_precision()
                    if global_ui_settings_from_config.get("type", "float").lower() == "float":
                        self.precisionSpinBox.setValue(global_ui_settings_from_config.get("precision", 1))
                    self.distributionComboBox.setCurrentText(
                        global_ui_settings_from_config.get("distribution", "uniform"))
                    self.addOutliersCheckBox.setChecked(global_ui_settings_from_config.get("add_outliers", False))
                    self.outlierPercentageSpinBox.setValue(
                        global_ui_settings_from_config.get("outlier_percentage", 5.0))
                    self.outlierMagnitudeSpinBox.setValue(global_ui_settings_from_config.get("outlier_magnitude", 3.0))

                default_settings_for_new = self._get_global_ui_settings()

                current_len_names = len(self.column_names)
                if num_cols_from_config > current_len_names:
                    self.column_names.extend(
                        [f"Столбец {i + 1}" for i in range(current_len_names, num_cols_from_config)])
                elif num_cols_from_config < current_len_names:
                    self.column_names = self.column_names[:num_cols_from_config]

                current_len_settings = len(self.column_settings)
                if num_cols_from_config > current_len_settings:
                    self.column_settings.extend(
                        [default_settings_for_new.copy() for _ in range(current_len_settings, num_cols_from_config)])
                elif num_cols_from_config < current_len_settings:
                    self.column_settings = self.column_settings[:num_cols_from_config]

                self.existing_settings_generation()
                self.statusbar.showMessage(f"Конфигурация загружена: {file_path}", 3000)
                self._mark_clean()
            except json.JSONDecodeError as e_json:
                QMessageBox.critical(self, "Ошибка загрузки", f"Файл конфигурации поврежден (JSON): {e_json}")
            except Exception as e_generic:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить конфигурацию: {e_generic}")

    def show_graph(self):
        """Отображает гистограмму по данным."""
        if self.generated_data.size == 0:
            QMessageBox.information(self, "Нет данных", "Таблица пуста. Нечего отображать.")
            return

        selected_indices = self.get_selected_indices()
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))

        bins_count = self.graphBinsSpinBox.value()

        if not selected_indices:
            data_to_plot = self.generated_data.flatten()
            data_to_plot = data_to_plot[~pd.isna(data_to_plot)]
            if data_to_plot.size == 0:
                QMessageBox.information(self, "Нет данных", "Нет числовых данных для построения графика.")
                plt.close()
                return
            plt.hist(data_to_plot, bins=bins_count, edgecolor="black", alpha=0.75, color='skyblue')
            plt.title("Распределение всех данных таблицы", fontsize=15)
        else:
            if len(selected_indices) == 1:
                idx = selected_indices[0]
                col_name = self.column_names[idx] if idx < len(self.column_names) else f"Столбец {idx + 1}"
                data_to_plot = self.generated_data[:, idx]
                data_to_plot = data_to_plot[~pd.isna(data_to_plot)]
                if data_to_plot.size == 0:
                    QMessageBox.information(self, "Нет данных", f"В столбце '{col_name}' нет числовых данных.")
                    plt.close()
                    return

                settings_info = self.column_settings[idx] if idx < len(self.column_settings) else {}
                plt.hist(data_to_plot, bins=bins_count, edgecolor="black", alpha=0.75, label=col_name, color='coral')

                title_string = f"Столбец: {col_name}"
                details_list = []
                if settings_info:
                    details_list.append(f"Тип: {settings_info.get('type', 'N/A')}")
                    details_list.append(f"Распред: {settings_info.get('distribution', 'N/A')}")
                if details_list:
                    title_string += f"\n({', '.join(details_list)})"
                plt.title(title_string, fontsize=14)
            else:
                num_colors = len(selected_indices)
                try:
                    colors = plt.cm.get_cmap('viridis', num_colors if num_colors > 0 else 1)
                except ValueError:
                    colors = lambda x: 'blue'

                for i, idx in enumerate(selected_indices):
                    col_name = self.column_names[idx] if idx < len(self.column_names) else f"Столбец {idx + 1}"
                    data_to_plot = self.generated_data[:, idx]
                    data_to_plot = data_to_plot[~pd.isna(data_to_plot)]
                    if data_to_plot.size > 0:
                        color_val = colors(i) if num_colors > 0 else colors(0)
                        plt.hist(data_to_plot, bins=bins_count, alpha=0.7, label=col_name, color=color_val)
                plt.title("Распределение данных для выбранных столбцов", fontsize=15)
                if len(selected_indices) > 1:
                    plt.legend()

        plt.xlabel("Значение", fontsize=12)
        plt.ylabel("Частота", fontsize=12)
        plt.tight_layout()
        plt.show()

    def clear_all_data_and_settings(self):
        """Очищает таблицу, имена, настройки и сбрасывает UI к дефолту."""
        msg_box_clear = QMessageBox(self)
        msg_box_clear.setWindowTitle('Очистить всё')
        msg_box_clear.setText("Вы уверены, что хотите удалить все данные и сбросить настройки?\n"
                              "Это действие необратимо.")
        msg_box_clear.setIcon(QMessageBox.Question)

        yes_btn_clear = msg_box_clear.addButton("да", QMessageBox.YesRole)
        no_btn_clear = msg_box_clear.addButton("отмена", QMessageBox.NoRole)
        msg_box_clear.setDefaultButton(no_btn_clear)
        msg_box_clear.setEscapeButton(no_btn_clear)

        msg_box_clear.exec_()

        if msg_box_clear.clickedButton() == yes_btn_clear:
            self.generated_data = np.array([])
            self.column_names = []
            self.column_settings = []

            self.dataTable.setRowCount(0)
            self.dataTable.setColumnCount(0)
            self.dataTable.setHorizontalHeaderLabels([])

            self._set_initial_ui_values()
            self.toggle_precision()

            self.statusbar.showMessage("Таблица и все настройки очищены.", 3000)
            self._mark_clean()

    def update_table_widget(self):
        """Обновляет QTableWidget данными."""
        is_completely_empty = self.generated_data.size == 0 and not self.column_names

        if is_completely_empty:
            self.dataTable.setRowCount(0)
            self.dataTable.setColumnCount(0)
            self.dataTable.setHorizontalHeaderLabels([])
            return

        expected_num_cols = len(self.column_names) if self.column_names else self.columnsSpinBox.value()
        expected_num_rows = self.quantitySpinBox.value()
        if self.generated_data.size > 0:
            expected_num_rows = self.generated_data.shape[0]

        self.dataTable.setRowCount(expected_num_rows)
        self.dataTable.setColumnCount(expected_num_cols)

        table_headers = self.column_names if self.column_names else [f"Столбец {j + 1}" for j in
                                                                     range(expected_num_cols)]
        self.dataTable.setHorizontalHeaderLabels(table_headers)

        if self.generated_data.size == 0:
            return

        actual_rows, actual_cols_in_data = self.generated_data.shape

        if actual_cols_in_data != expected_num_cols and expected_num_cols > 0:
            return

        for i in range(actual_rows):
            for j in range(actual_cols_in_data):
                value = self.generated_data[i, j]

                if pd.isna(value):
                    item_text_to_display = "NaN"
                else:
                    col_setting_cell = self.column_settings[j] if j < len(self.column_settings) else None

                    if col_setting_cell and col_setting_cell["type"].lower() == "integer":
                        item_text_to_display = str(int(round(value)))
                    elif col_setting_cell and col_setting_cell["type"].lower() == "float":
                        precision_for_cell = col_setting_cell.get('precision', 1)
                        try:
                            item_text_to_display = f"{float(value):.{precision_for_cell}f}"
                        except (ValueError, TypeError):
                            item_text_to_display = str(value)
                    else:
                        item_text_to_display = str(value)

                self.dataTable.setItem(i, j, QTableWidgetItem(item_text_to_display))

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Обработка закрытия окна с предложением сохранить конфигурацию."""
        proceed_to_close = False
        if self.is_dirty:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('Несохраненные изменения')
            msg_box.setText("Есть несохраненные изменения в конфигурации. Сохранить их перед выходом?")
            msg_box.setIcon(QMessageBox.Question)

            save_button = msg_box.addButton("сохранить", QMessageBox.AcceptRole)
            discard_button = msg_box.addButton("не сохранять", QMessageBox.DestructiveRole)
            cancel_button = msg_box.addButton("отмена", QMessageBox.RejectRole)

            msg_box.setDefaultButton(save_button)
            msg_box.setEscapeButton(cancel_button)
            msg_box.exec_()

            clicked_button = msg_box.clickedButton()
            if clicked_button == save_button:
                self.save_configuration()
                if not self.is_dirty:
                    proceed_to_close = True
            elif clicked_button == discard_button:
                proceed_to_close = True
        else:
            proceed_to_close = True

        if proceed_to_close:
            exit_msg_box = QMessageBox(self)
            exit_msg_box.setWindowTitle('Выход из NumHeaven')
            exit_msg_box.setText("Вы уверены, что хотите закрыть программу?")
            exit_msg_box.setIcon(QMessageBox.Question)

            yes_button = exit_msg_box.addButton("да", QMessageBox.YesRole)
            no_button = exit_msg_box.addButton("нет", QMessageBox.NoRole)

            exit_msg_box.setDefaultButton(no_button)
            exit_msg_box.setEscapeButton(no_button)

            exit_msg_box.exec_()

            if exit_msg_box.clickedButton() == yes_button:
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    ex.show()
    sys.exit(app.exec_())
