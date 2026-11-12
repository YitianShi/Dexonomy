"""Qt desktop interface for browsing generated grasps."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QCompleter, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

if __package__:
    from .grasp_viewer_panel import (
        ALL_TYPES, DATASET_LABELS, SAMPLING, build_viewer_command,
        group_taxonomies, read_group_details,
    )
else:
    from grasp_viewer_panel import (
        ALL_TYPES, DATASET_LABELS, SAMPLING, build_viewer_command,
        group_taxonomies, read_group_details,
    )


STYLES = """
QMainWindow, QWidget#page { background: #f3f5f7; }
QWidget { color: #182b32; font-size: 13px; }
QLabel { background: transparent; }
QLabel#brand { font-size: 12px; font-weight: 700; letter-spacing: 3px; color: #0d766e; }
QLabel#heading { font-size: 28px; font-weight: 700; }
QLabel#muted { color: #73838b; font-size: 12px; }
QLabel#section { font-size: 14px; font-weight: 600; }
QLabel#previewTitle { font-size: 21px; font-weight: 600; }
QLabel#badge { color: #087c6e; background: #e5f4ef; border-radius: 11px; padding: 4px 12px; }
QFrame#card { background: white; border: 1px solid #e2e8eb; border-radius: 14px; }
QFrame#metric { background: #f6f8fa; border-radius: 10px; }
QLabel#metricValue { font-size: 20px; font-weight: 600; }
QComboBox, QSpinBox { background: #f9fafb; border: 1px solid #dce3e7; border-radius: 7px;
 padding: 8px 11px; selection-background-color: #d6eee8; selection-color: #143f38; min-height: 20px; }
QComboBox:hover, QSpinBox:hover { border-color: #a8bcbf; }
QComboBox:focus, QSpinBox:focus { border: 1px solid #0e8c7e; }
QComboBox:disabled { color: #60737b; background: #f1f4f5; border-color: #e6ebee; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: none; }
QComboBox QAbstractItemView { background: white; border: 1px solid #dbe4e7;
 outline: none; padding: 4px; selection-background-color: #e2f3ee; selection-color: #15584b; }
QAbstractItemView::item { min-height: 29px; padding: 4px; }
QPushButton { background: white; border: 1px solid #d9e2e6; border-radius: 8px;
 padding: 11px 16px; font-weight: 600; }
QPushButton:hover { background: #f0f7f5; border-color: #9abbb3; }
QPushButton#primary { color: white; background: #087f72; border: none; }
QPushButton#primary:hover { background: #06695e; }
QPushButton#primary:disabled { background: #a9c4bf; color: #f1f7f5; }
QToolButton { background: transparent; border: none; padding: 8px 0; color: #71848b; }
QToolButton:hover { color: #087f72; }
QSlider::groove:horizontal { height: 4px; background: #e0e8e9; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #128b7b; border-radius: 2px; }
QSlider::handle:horizontal { background: white; border: 2px solid #128b7b;
 width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }
QPlainTextEdit { background: #f6f8fa; color: #4b606a; border: 1px solid #e2e8eb;
 border-radius: 8px; padding: 8px; font-family: 'DejaVu Sans Mono'; font-size: 11px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #c9d5d9; border-radius: 3px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def label(text, role=None):
    widget = QLabel(text)
    if role:
        widget.setObjectName(role)
    return widget


def card(title):
    widget = QFrame()
    widget.setObjectName("card")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 18, 20, 20)
    layout.setSpacing(8)
    layout.addWidget(label(title, "section"))
    return widget, layout


class Picker(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#6e858c"), 1.5))
        x, y = self.width()-17, self.height()/2
        painter.drawPolyline(QPolygonF([QPointF(x-4, y-2), QPointF(x, y+2), QPointF(x+4, y-2)]))


def searchable_combo():
    widget = Picker()
    widget.setEditable(True)
    widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    widget.setMaxVisibleItems(16)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    completer = QCompleter(widget.model(), widget)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    widget.setCompleter(completer)
    return widget


class ModelPreview(QWidget):
    """CPU-rendered mesh preview: independent of simulation and display sampling."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.geometries = []
        self.center = np.zeros(3)
        self.radius = 1.0
        self.yaw, self.elevation, self.zoom = -55.0, 26.0, 1.0
        self.drag_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_details(self, details):
        self.geometries = details["geometries"]
        vertices = np.concatenate([vertices for vertices, _ in self.geometries])
        self.center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        self.radius = max(float(np.max(np.ptp(vertices, axis=0))), .001)
        self.floor = float(vertices[:, 2].min())
        self.zoom = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f4f7f9"))
        if not self.geometries:
            painter.setPen(QColor("#94a4ad"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择物体或场景以查看预览")
            return
        yaw, el = np.deg2rad([self.yaw, self.elevation])
        basis = np.array([[-np.sin(yaw), np.cos(yaw), 0],
                          [-np.sin(el)*np.cos(yaw), -np.sin(el)*np.sin(yaw), np.cos(el)],
                          [np.cos(el)*np.cos(yaw), np.cos(el)*np.sin(yaw), np.sin(el)]])
        factor = min(self.width()*.64, self.height()*.82) / self.radius * self.zoom
        origin = np.array([self.width()/2, self.height()*.50])

        def project(points):
            coords = (points - self.center) @ basis.T
            return coords[:, :2] * [factor, -factor] + origin

        # A restrained floor grid provides orientation without obscuring the model.
        painter.setPen(QPen(QColor("#dfe7eb"), .7))
        extent = self.radius*.85
        for value in np.linspace(-extent, extent, 11):
            for axis in (0, 1):
                points = np.array([[-extent, value, self.floor], [extent, value, self.floor]])
                if axis:
                    points[:, [0, 1]] = points[:, [1, 0]]
                points[:, :2] += self.center[:2]
                xy = project(points)
                painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))
        floor_xy = project(np.array([[*self.center[:2], self.floor]]))[0]
        shadow = QRadialGradient(QPointF(*floor_xy), min(self.width(), self.height())*.30)
        shadow.setColorAt(0, QColor(37, 61, 68, 32))
        shadow.setColorAt(1, QColor(37, 61, 68, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow)
        painter.drawEllipse(QPointF(*floor_xy), self.width()*.31, self.height()*.12)

        colors = np.array([[83, 151, 143], [145, 163, 173], [194, 162, 126],
                           [118, 139, 168], [167, 151, 168], [128, 163, 146]], dtype=float)
        faces_to_draw = []
        light = np.array([-.35, -.45, .82]);light /= np.linalg.norm(light)
        for index, (vertices, faces) in enumerate(self.geometries):
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            shades = .63 + .37*np.abs(normals @ light)
            depths = ((triangles.mean(axis=1)-self.center) @ basis[2])
            xy = project(triangles.reshape(-1, 3)).reshape(-1, 3, 2)
            base = colors[index % len(colors)]
            faces_to_draw.extend((depth, points, base*shade)
                                 for depth, points, shade in zip(depths, xy, shades))
        for _, points, color in sorted(faces_to_draw, key=lambda item: item[0]):
            fill = QColor(*(int(value) for value in color))
            painter.setBrush(fill)
            painter.setPen(QPen(fill, .45))
            painter.drawPolygon(QPolygonF([QPointF(*point) for point in points]))
        painter.setPen(QColor("#91a0a7"))
        painter.drawText(18, self.height()-16, "拖动旋转 · 滚轮缩放")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag_position is not None:
            delta = event.position()-self.drag_position
            self.drag_position = event.position()
            self.yaw -= delta.x()*.45
            self.elevation = float(np.clip(self.elevation+delta.y()*.35, -75, 85))
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * 1.12**(event.angleDelta().y()/120), .3, 3))
        self.update()


class ViewerControlPanel(QMainWindow):
    def __init__(self, args, collect_groups):
        super().__init__()
        self.args, self.collect_groups = args, collect_groups
        catalog = json.loads(args.catalog.read_text())
        self.datasets = {entry["id"]: {**entry, "root": (args.catalog.parent/entry["data"]).resolve()}
                         for entry in catalog["datasets"]}
        self.initial = args.dataset if args.dataset in self.datasets else next(iter(self.datasets))
        if Path(args.root).resolve() != self.datasets[self.initial]["root"]:
            self.initial = "custom"
            self.datasets["custom"] = {"id": "custom", "root": Path(args.root).resolve()}
        self.events = queue.Queue()
        self.cache = {}
        self.details_cache = OrderedDict()
        self.details_token = 0
        self.process = self.pending_command = self.stop_deadline = None
        self.loading = True
        self.current_group = None
        self._build_widgets()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(80)
        self._request_dataset()

    def _build_widgets(self):
        self.setWindowTitle("Dexonomy · Grasp Explorer")
        self.resize(1240, 910)
        self.setMinimumSize(1040, 790)
        self.setStyleSheet(STYLES)
        page = QWidget();page.setObjectName("page");self.setCentralWidget(page)
        outer = QVBoxLayout(page);outer.setContentsMargins(30, 24, 30, 18);outer.setSpacing(20)
        header = QHBoxLayout()
        title = QVBoxLayout();title.setSpacing(4)
        title.addWidget(label("DEXONOMY  /  GRASP EXPLORER", "brand"))
        title.addWidget(label("抓取可视化", "heading"))
        title.addWidget(label("浏览物体，选择抓取，再打开三维视图。", "muted"))
        header.addLayout(title);header.addStretch()
        self.badge = label("正在读取", "badge");header.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)
        body = QHBoxLayout();body.setSpacing(20);outer.addLayout(body, 1)
        left_column = QWidget();left_column.setFixedWidth(374)
        left_shell = QVBoxLayout(left_column);left_shell.setContentsMargins(0, 0, 0, 0);left_shell.setSpacing(10)
        body.addWidget(left_column)
        controls = QWidget()
        left = QVBoxLayout(controls);left.setContentsMargins(0, 0, 0, 0);left.setSpacing(16)
        scroll = QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls)
        left_shell.addWidget(scroll, 1)

        data_card, data = card("01   选择数据")
        self.dataset = Picker()
        for key in self.datasets:
            self.dataset.addItem(DATASET_LABELS.get(key, "自定义目录"), key)
        self.dataset.setCurrentIndex(self.dataset.findData(self.initial))
        data.addWidget(self.dataset)
        self.inventory_label = label("正在读取可用条目…", "muted");data.addWidget(self.inventory_label)
        data.addWidget(label("抓取类型"))
        self.type_picker = searchable_combo();data.addWidget(self.type_picker)
        self.object_label = label("物体");data.addWidget(self.object_label)
        self.item_picker = searchable_combo();data.addWidget(self.item_picker)
        self.scale_label = label("物体尺度");data.addWidget(self.scale_label)
        self.scale_picker = Picker();data.addWidget(self.scale_picker)
        self.scale_hint = label("", "muted");self.scale_hint.setWordWrap(True);data.addWidget(self.scale_hint)
        left.addWidget(data_card)
        self.dataset.currentIndexChanged.connect(self._request_dataset)
        self.type_picker.activated.connect(self._refresh_items)
        self.item_picker.activated.connect(self._select_object)
        self.item_picker.lineEdit().editingFinished.connect(self._select_object)
        self.scale_picker.currentIndexChanged.connect(self._select_scale)

        sample_card, sample = card("02   采样与显示")
        self.sampling = Picker()
        for text, value in SAMPLING.items():self.sampling.addItem(text, value)
        self.sampling.setCurrentIndex(self.sampling.findData(self.args.sampling))
        sample.addWidget(self.sampling)
        numbers = QGridLayout();numbers.setHorizontalSpacing(14)
        numbers.addWidget(label("每物体最多 / 个"), 0, 0);numbers.addWidget(label("随机种子"), 0, 1)
        self.count = QSpinBox();self.count.setRange(0, 1000000);self.count.setValue(self.args.max_grasps)
        self.seed = QSpinBox();self.seed.setRange(0, 2147483647);self.seed.setValue(self.args.sample_seed)
        self.count.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.seed.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        numbers.addWidget(self.count, 1, 0);numbers.addWidget(self.seed, 1, 1);sample.addLayout(numbers)
        sample.addWidget(label("0 表示全部；场景内每个物体分别采样。", "muted"))
        more = QToolButton();more.setText("▸ 更多显示设置 · 姿态 / 透明度");more.setCheckable(True)
        sample.addWidget(more)
        display_options = QWidget();display_options.hide()
        display_layout = QVBoxLayout(display_options);display_layout.setContentsMargins(0, 0, 0, 0);display_layout.setSpacing(8)
        more.toggled.connect(lambda checked:(display_options.setVisible(checked),more.setText(
            "▾ 更多显示设置 · 姿态 / 透明度" if checked else "▸ 更多显示设置 · 姿态 / 透明度")))
        self.phase = Picker()
        for text, value in [("抓取姿态 · Grasp", "grasp"), ("预抓取 · Pregrasp", "pregrasp"), ("挤压姿态 · Squeeze", "squeeze")]:
            self.phase.addItem(text, value)
        self.phase.setCurrentIndex(self.phase.findData(self.args.phase));display_layout.addWidget(self.phase)
        opacity = QHBoxLayout();opacity.addWidget(label("手部不透明度"));opacity.addStretch()
        self.alpha_label = label(f"{self.args.hand_alpha:.0%}", "muted");opacity.addWidget(self.alpha_label);display_layout.addLayout(opacity)
        self.alpha = QSlider(Qt.Orientation.Horizontal);self.alpha.setRange(0, 100);self.alpha.setValue(round(self.args.hand_alpha*100))
        self.alpha.valueChanged.connect(lambda value:self.alpha_label.setText(f"{value}%"));display_layout.addWidget(self.alpha)
        sample.addWidget(display_options)
        left.addWidget(sample_card)
        self.open_button = QPushButton("打开 / 更新抓取可视化  ↗");self.open_button.setObjectName("primary")
        self.open_button.setEnabled(False);self.open_button.clicked.connect(self.open_viewer);left_shell.addWidget(self.open_button)
        stop = QPushButton("关闭 MuJoCo 窗口");stop.clicked.connect(self.stop_viewer);left_shell.addWidget(stop)
        left.addStretch()

        right = QVBoxLayout();right.setSpacing(14);body.addLayout(right, 1)
        preview_card, preview = card("模型预览")
        self.preview_title = label("选择物体或场景", "previewTitle")
        self.preview_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        preview.addWidget(self.preview_title)
        self.preview_subtitle = label("", "muted");self.preview_subtitle.setWordWrap(True);preview.addWidget(self.preview_subtitle)
        self.preview = ModelPreview();preview.addWidget(self.preview, 1)
        metric_row = QHBoxLayout();metric_row.setSpacing(10);self.dimensions = []
        for axis in "XYZ":
            metric = QFrame();metric.setObjectName("metric");layout = QVBoxLayout(metric);layout.setContentsMargins(14, 10, 14, 12)
            layout.addWidget(label(f"{axis} 轴尺寸", "muted"))
            value = label("—", "metricValue");layout.addWidget(value);self.dimensions.append(value);metric_row.addWidget(metric, 1)
        preview.addLayout(metric_row)
        self.dimension_note = label("尺寸为模型坐标系下的包围盒，单位厘米。", "muted");preview.addWidget(self.dimension_note)
        preview.addWidget(label("抓取姿态在 MuJoCo 窗口中显示。", "muted"));right.addWidget(preview_card, 1)
        self.log_toggle = QToolButton();self.log_toggle.setText("▸ 运行信息");self.log_toggle.setCheckable(True)
        right.addWidget(self.log_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.log = QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(220);self.log.setFixedHeight(140);self.log.hide()
        self.log_toggle.toggled.connect(lambda checked:(self.log.setVisible(checked),self.log_toggle.setText("▾ 运行信息" if checked else "▸ 运行信息")))
        right.addWidget(self.log)
        footer = QHBoxLayout()
        self.status = label("正在读取数据目录…", "muted");self.status.setWordWrap(True);footer.addWidget(self.status, 1)
        footer.addWidget(label("MuJoCo  ·  1/2/3 切换姿态   H 隐藏手   Q 退出", "muted"));outer.addLayout(footer)
        screen = self.screen().availableGeometry()
        self.move(screen.center()-self.rect().center())

    def _request_dataset(self):
        key = self.dataset.currentData()
        self.loading = True;self.current_group = None;self.open_button.setEnabled(False)
        self.details_token += 1
        self.status.setText("正在读取物体、场景与抓取类型…");self.badge.setText("正在读取")
        if key in self.cache:
            self._apply_inventory(key)
            return
        def scan():
            options = copy.copy(self.args)
            options.root = str(self.datasets[key]["root"])
            options.taxonomy, options.scene, options.object, options.start, options.limit = "all", "", "*", 0, 0
            try:
                groups, _ = self.collect_groups(options);self.events.put(("inventory", (key, groups, None)))
            except Exception as exc:self.events.put(("inventory", (key, [], str(exc))))
        threading.Thread(target=scan, daemon=True).start()

    def _apply_inventory(self, key):
        if key != self.dataset.currentData():return
        groups = self.cache[key]
        types = set().union(*(group_taxonomies(group) for group in groups))
        self.type_picker.blockSignals(True);previous = self.type_picker.currentData()
        self.type_picker.clear();self.type_picker.addItem(ALL_TYPES, "all")
        for name in sorted(types, key=lambda value:int(value.split("_", 1)[0])):
            number, text = name.split("_", 1)
            self.type_picker.addItem(f"{int(number):02d} · {text.replace('_', ' ')}", name)
        requested = previous or self.args.taxonomy
        self.type_picker.setCurrentIndex(max(0, self.type_picker.findData(requested)))
        self.type_picker.blockSignals(False)
        self.loading = False
        single = bool(groups and groups[0].samples is not None)
        count = len({group.name.split('/')[0] for group in groups}) if single else len(groups)
        self.inventory_label.setText(f"{count:,} 个{'物体' if single else '含抓取的场景'}  ·  {len(types)} 类抓取")
        self.object_label.setText("物体  ·  可输入编号搜索" if single else "场景  ·  可输入编号搜索")
        for widget in (self.scale_label, self.scale_picker, self.scale_hint):widget.setVisible(single)
        self._refresh_items()

    def _refresh_items(self):
        if self.loading:return
        taxonomy = self.type_picker.currentData()
        if taxonomy is None:return
        self.filtered = [group for group in self.cache[self.dataset.currentData()]
                         if taxonomy == "all" or taxonomy in group_taxonomies(group)]
        previous = self.item_picker.currentData()
        self.item_picker.blockSignals(True);self.item_picker.clear()
        single = bool(self.filtered and self.filtered[0].samples is not None)
        names = sorted({group.name.split('/')[0] if single else group.name for group in self.filtered})
        for name in names:self.item_picker.addItem(name, name)
        preferred = previous or (self.args.object if single and self.args.object != '*' else self.args.scene)
        self.item_picker.setCurrentIndex(max(0, self.item_picker.findData(preferred)))
        self.item_picker.blockSignals(False)
        self._select_object()

    def _select_object(self):
        if self.loading:return
        name = self.item_picker.currentData()
        if name is None or self.item_picker.currentText() != str(name):
            self.current_group = None;self.open_button.setEnabled(False)
            self.status.setText("请从搜索结果中选择一个物体或场景。")
            return
        groups = [group for group in self.filtered if group.name == name or group.name.startswith(name+'/')]
        self.details_token += 1;token = self.details_token;key = self.dataset.currentData()
        self.current_group = None;self.open_button.setEnabled(False);self.badge.setText("读取模型")
        self.scale_picker.blockSignals(True);self.scale_picker.clear();self.scale_picker.addItem("正在读取实际尺度…")
        self.scale_picker.setEnabled(False);self.scale_picker.blockSignals(False)
        self.status.setText("正在读取实际尺度与模型尺寸…")
        def load():
            try:
                rows = [(group, self.details_cache.get((key, group.name)) or read_group_details(group)) for group in groups]
                self.events.put(("details", (key, token, rows, None)))
            except Exception as exc:self.events.put(("details", (key, token, [], str(exc))))
        threading.Thread(target=load, daemon=True).start()

    def _apply_details(self, key, token, rows, error):
        for group, details in rows:
            self.details_cache[(key, group.name)] = details
            self.details_cache.move_to_end((key, group.name))
            while len(self.details_cache)>8:self.details_cache.popitem(last=False)
        if token != self.details_token or key != self.dataset.currentData():return
        if error:
            self.status.setText(f"模型读取失败：{error}");self.badge.setText("读取失败")
            return
        self.scale_picker.blockSignals(True);self.scale_picker.clear()
        rows = sorted(rows, key=lambda row:float(np.prod(row[1]['scale'])) if row[1]['single'] else 0)
        for group, details in rows:self.scale_picker.addItem(details['scale_label'], (group, details))
        index = next((i for i, (group, _) in enumerate(rows) if group.name == self.args.scene), 0)
        self.scale_picker.setCurrentIndex(index);self.scale_picker.setEnabled(len(rows)>1)
        self.scale_picker.blockSignals(False)
        self.scale_hint.setText("该物体仅保存了这一种尺度。" if len(rows)==1 else "选择已保存的实际缩放倍数。")
        self._select_scale()

    def _select_scale(self):
        value = self.scale_picker.currentData()
        if value is None:return
        group, details = value;self.current_group = group
        title = (f"物体 {details['target']}" if details['single'] else group.name.replace('scene_', '场景 '))
        if len(title)>38:
            self.preview_title.setText(title[:18]+'…'+title[-10:]);self.preview_title.setToolTip(title)
        else:self.preview_title.setText(title);self.preview_title.setToolTip('')
        self.preview_subtitle.setText(details['scale_label'] if details['single'] else f"{details['object_count']} 个物体 · 使用场景中保存的位置和尺度")
        for widget, size in zip(self.dimensions, details['dimensions']):widget.setText(f"{size*100:.1f} cm")
        self.dimension_note.setText("尺寸为模型坐标系下的包围盒，单位厘米。" if details['single'] else "尺寸为场景中所有物体的包围盒，单位厘米。")
        self.preview.set_details(details);self.open_button.setEnabled(True);self.badge.setText("就绪")
        self.status.setText("设置采样参数后，点击“打开 / 更新抓取可视化”。")

    def open_viewer(self):
        if self.current_group is None:return
        try:
            command = build_viewer_command(
                self.args, self.datasets[self.dataset.currentData()]['root'], self.current_group.name,
                self.type_picker.currentData(), self.count.value(), self.seed.value(),
                self.sampling.currentData(), self.phase.currentData(), self.alpha.value()/100,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "检查显示参数", str(exc));return
        if self.process is not None and self.process.poll() is None:
            self.pending_command = command;self.process.terminate();self.stop_deadline = time.monotonic()+3
            self.status.setText("正在更新 MuJoCo 可视化…")
        else:self._start_viewer(command)

    def _start_viewer(self, command):
        self.pending_command = self.stop_deadline = None
        self.log.appendPlainText(f"\n打开 {self.dataset.currentText()} / {self.item_picker.currentText()}")
        environment = {**os.environ, 'MUJOCO_GL':'glfw', 'PYTHONUNBUFFERED':'1',
                       'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        try:
            self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:self.status.setText(f"启动失败：{exc}");return
        process = self.process;self.badge.setText("MuJoCo 运行中")
        self.status.setText("正在加载抓取；可在运行信息中查看实际样本数。")
        def read_output():
            with process.stdout:
                for line in process.stdout:self.events.put(('log', line))
        threading.Thread(target=read_output, daemon=True).start()

    def stop_viewer(self):
        self.pending_command = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate();self.stop_deadline = time.monotonic()+3
            self.status.setText("正在关闭 MuJoCo 窗口…")

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log':self.log.appendPlainText(value.rstrip('\n'))
                elif kind == 'details':self._apply_details(*value)
                else:
                    key, groups, error = value
                    if error:
                        if key == self.dataset.currentData():self.status.setText(f"目录读取失败：{error}")
                    else:self.cache[key] = groups;self._apply_inventory(key)
        except queue.Empty:pass
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.process = self.stop_deadline = None
                if self.pending_command is not None:
                    command = self.pending_command;self._start_viewer(command)
                else:
                    self.badge.setText("就绪")
                    self.status.setText("可视化已关闭，可选择下一组抓取。" if code in (0,-15)
                                        else f"可视化退出（代码 {code}），请查看运行信息。")
                    if code not in (0,-15):self.log_toggle.setChecked(True)
            elif self.stop_deadline is not None and time.monotonic()>self.stop_deadline:
                self.process.kill();self.stop_deadline = None

    def closeEvent(self, event):
        self.timer.stop();self.stop_viewer()
        if self.process is not None:
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        event.accept()


def launch_panel(args, collect_groups):
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC', 10))
    panel = ViewerControlPanel(args, collect_groups)
    panel.show()
    return app.exec()
