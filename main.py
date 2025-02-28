import os
import sys
import json
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QInputDialog
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QGuiApplication

from main_ui import Ui_MainWindow  # Import the generated UI class

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
QGuiApplication.setAttribute(2, True)

class MainWidget(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup_connections()

        self.column_names = []
        self.column_settings = []


    def setup_connections(self):
        self.generateButton.clicked.connect(self.generate_data)
        self.saveButton.clicked.connect(self.save_to_csv)
        self.showButton.clicked.connect(self.show_graph)
        self.dataTypeComboBox.currentTextChanged.connect(self.toggle_precision)
        self.applyButton.clicked.connect(self.apply_column_settings)
        self.dataTable.itemSelectionChanged.connect(self.update_column_settings_ui)
        self.searchPushButton.clicked.connect(self.search_column)
        self.saveConfigButton.clicked.connect(self.save_configuration)
        self.loadConfigButton.clicked.connect(self.load_configuration)
        self.resetButton.clicked.connect(self.reset_column_settings)


    def toggle_precision(self):
        """Enable/Disable Precision SpinBox based on number type"""
        if self.dataTypeComboBox.currentText() == "integer":
            self.precisionSpinBox.setDisabled(True)
            self.precisionSpinBox.setStyleSheet("border: 1px solid gray;")
        else:
            self.precisionSpinBox.setDisabled(False)
            self.precisionSpinBox.setStyleSheet("border: 1px solid green;")

    def update_table(self):
        """Display generated data in the QTableWidget"""
        if not hasattr(self, 'generated_data'):
            return

        rows, cols = self.generated_data.shape
        self.dataTable.setRowCount(rows)
        self.dataTable.setColumnCount(cols)
        self.dataTable.setHorizontalHeaderLabels(self.column_names)

        self.dataTable.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)

        for i in range(rows):
            for j in range(cols):
                value = self.generated_data[i, j]
                if self.column_settings[j]["type"] == "integer":
                    value = int(value)
                item = QTableWidgetItem(str(value))
                self.dataTable.setItem(i, j, item)

    def update_column_settings_ui(self):
        """Load settings of the first selected column into the UI"""
        selected_columns = self.get_selected_columns()

        if not selected_columns:
            return

        first_col = selected_columns[0]
        settings = self.column_settings[first_col]

        self.minSpinBox.setValue(settings["min"])
        self.maxSpinBox.setValue(settings["max"])
        self.dataTypeComboBox.setCurrentText(settings["type"])
        self.precisionSpinBox.setValue(settings["precision"])
        self.distributionComboBox.setCurrentText(settings["distribution"])

        print(f"Loaded settings for Column {first_col + 1}")

    def apply_column_settings(self):
        """Apply current spin box settings to selected columns only"""
        selected_columns = self.get_selected_columns()

        if not selected_columns:
            print("No columns selected for applying settings!")
            return

        new_settings = {
            "min": self.minSpinBox.value(),
            "max": self.maxSpinBox.value(),
            "type": self.dataTypeComboBox.currentText(),
            "precision": self.precisionSpinBox.value(),
            "distribution": self.distributionComboBox.currentText()
        }

        for col in selected_columns:
            self.column_settings[col] = new_settings.copy()

            min_val, max_val, dtype, precision, distribution = new_settings.values()
            if dtype == "float":
              if distribution == "uniform":
                self.generated_data[:, col] = np.random.uniform(min_val, max_val, self.generated_data.shape[0])
              elif distribution == "normal":
                mean = (max_val + min_val) / 2
                std_dev = (max_val - min_val) / 4
                self.generated_data[:, col] = np.random.normal(mean, std_dev, self.generated_data.shape[0])
              elif distribution == "exponential":
                scale = (max_val - min_val) / 2
                self.generated_data[:,col] = np.random.exponential(scale, self.generated_data.shape[0]) + min_val
                self.generated_data[:, col] = np.clip(self.generated_data[:, col], min_val, max_val)


              self.generated_data[:, col] = np.round(self.generated_data[:, col], precision)
            elif dtype == "integer":
              if distribution == "uniform":
                self.generated_data[:, col] = np.random.randint(min_val, max_val + 1, self.generated_data.shape[0])
              elif distribution == "normal":
                  mean = (max_val + min_val) / 2
                  std_dev = (max_val - min_val) / 4
                  self.generated_data[:, col] = np.round(np.random.normal(mean, std_dev, self.generated_data.shape[0])).astype(int)
                  self.generated_data[:, col] = np.clip(self.generated_data[:,col], min_val, max_val)
              elif distribution == "exponential":
                  scale = (max_val - min_val) / 2  # Approximation
                  self.generated_data[:, col] = np.round(np.random.exponential(scale, self.generated_data.shape[0]) + min_val).astype(int)
                  self.generated_data[:, col] = np.clip(self.generated_data[:, col], min_val, max_val)

        self.update_table()
        print(f"Applied settings to columns: {selected_columns}")

    def get_selected_columns(self):
        """Returns a list of selected column indices"""
        selected_indexes = self.dataTable.selectedIndexes()
        return list(set(index.column() for index in selected_indexes))

    def rename_column(self, index):
        """Rename only the double-clicked column and ensure only ONE pop-up appears."""
        self.dataTable.horizontalHeader().sectionDoubleClicked.disconnect()

        old_name = self.column_names[index]
        new_name, ok = QInputDialog.getText(self, "Rename Column", f"Enter new name for '{old_name}':")

        if ok and new_name.strip():
            self.column_names[index] = new_name.strip()
            self.update_table()
            print(f"Renamed Column {index + 1} to '{new_name}'")

        self.dataTable.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)

    def search_column(self):
        """Highlight columns that match the search term"""
        search_text = self.searchLineEdit.text().strip().lower()

        if not search_text:
            return

        for col, name in enumerate(self.column_names):
            if search_text in name.lower():
                self.dataTable.selectColumn(col)
                print(f"Found column: {name} (Index {col})")
                return

        print("No matching column found!")

    def generate_data(self):
        """Generate data using saved settings per column"""
        num_columns = max(1, self.columnsSpinBox.value())
        quantity = self.quantitySpinBox.value()

        if not hasattr(self, 'column_names') or len(self.column_names) != num_columns:
            self.column_names = [f"Column {i + 1}" for i in range(num_columns)]

        if not hasattr(self, 'column_settings') or len(self.column_settings) != num_columns:
            self.column_settings = [{
                "min": self.minSpinBox.value(),
                "max": self.maxSpinBox.value(),
                "type": self.dataTypeComboBox.currentText(),
                "precision": self.precisionSpinBox.value(),
                "distribution" : self.distributionComboBox.currentText()
            } for _ in range(num_columns)]

        data = []
        for col, settings in enumerate(self.column_settings):
            if settings["min"] >= settings["max"]:
                print(f"Error: Min must be less than Max in Column {col + 1}!")
                return

            min_val = settings["min"]
            max_val = settings["max"]
            dtype = settings["type"]
            precision = settings["precision"]
            distribution = settings["distribution"]

            if dtype == "integer":
              if distribution == "uniform":
                column_data = np.random.randint(min_val, max_val + 1, quantity)
              elif distribution == "normal":
                  mean = (max_val + min_val) / 2
                  std_dev = (max_val - min_val) / 4
                  column_data = np.round(np.random.normal(mean, std_dev, quantity)).astype(int)
                  column_data = np.clip(column_data, min_val, max_val)
              elif distribution == "exponential":
                  scale = (max_val - min_val) / 2
                  column_data = np.round(np.random.exponential(scale, quantity) + min_val).astype(int)
                  column_data = np.clip(column_data, min_val, max_val)

            elif dtype == "float":
              if distribution == "uniform":
                column_data = np.random.uniform(min_val, max_val, quantity)
              elif distribution == "normal":
                mean = (max_val + min_val) / 2
                std_dev = (max_val - min_val) / 4
                column_data = np.random.normal(mean, std_dev, quantity)
              elif distribution == "exponential":
                scale = (max_val - min_val) / 2
                column_data = np.random.exponential(scale, quantity) + min_val
                column_data = np.clip(column_data, min_val, max_val)

              column_data = np.round(column_data, precision)

            data.append(column_data)

        self.generated_data = np.column_stack(data)
        self.update_table()

    def reset_column_settings(self):
        """Reset all column settings to match the current global UI values"""
        num_columns = self.columnsSpinBox.value()

        default_settings = {
            "min": self.minSpinBox.value(),
            "max": self.maxSpinBox.value(),
            "type": self.dataTypeComboBox.currentText(),
            "precision": self.precisionSpinBox.value(),
            "distribution": self.distributionComboBox.currentText()
        }

        self.column_settings = [default_settings.copy() for _ in range(num_columns)]
        self.generate_data()
        print("Reset all column settings to default.")

    def save_to_csv(self):
        if not hasattr(self, 'generated_data'):
            print("No data to save! Generate some first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)")

        if file_path:
            df = pd.DataFrame(self.generated_data)

            for col in range(df.shape[1]):
                if self.column_settings[col]["type"].lower() == "integer":
                    df[col] = df[col].astype(int)

            df.columns = self.column_names
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")

    def save_configuration(self):
        """Save the current configuration to a file"""
        config = {
            "num_columns": self.columnsSpinBox.value(),
            "column_settings": self.column_settings,
            "column_names": self.column_names,
            "quantity": self.quantitySpinBox.value(),
            "min": self.minSpinBox.value(),
            "max": self.maxSpinBox.value(),
            "precision": self.precisionSpinBox.value(),
            "type": self.dataTypeComboBox.currentText(),
            "distribution": self.distributionComboBox.currentText()
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)")

        if file_path:
            with open(file_path, "w") as config_file:
                json.dump(config, config_file, indent=4)
            print(f"Configuration saved to {file_path}")

    def load_configuration(self):
        """Load configuration from a file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Configuration", "", "JSON Files (*.json);;All Files (*)")

        if file_path:
            with open(file_path, "r") as config_file:
                config = json.load(config_file)

            self.columnsSpinBox.setValue(config.get("num_columns", 1))
            self.column_settings = config.get("column_settings", [])
            self.column_names = config.get("column_names", [])
            self.quantitySpinBox.setValue(config.get("quantity", 100))
            self.minSpinBox.setValue(config.get("min", 0))
            self.maxSpinBox.setValue(config.get("max", 100))
            self.precisionSpinBox.setValue(config.get("precision", 0))
            self.dataTypeComboBox.setCurrentText(config.get("type", "float"))
            self.distributionComboBox.setCurrentText(config.get("distribution", "uniform"))

            self.generate_data()
            print(f"Configuration loaded from {file_path}")

    def show_graph(self):
        if not hasattr(self, 'generated_data'):
            print("No data to visualize! Generate some first.")
            return

        if self.generated_data.size == 0:
            print("No data to display (empty array).")
            return

        plt.figure(figsize=(8, 5))

        flat_data = self.generated_data.flatten()

        plt.hist(flat_data, bins=30, edgecolor="black", alpha=0.75)
        plt.title("Generated Data Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    ex.show()
    sys.exit(app.exec())
