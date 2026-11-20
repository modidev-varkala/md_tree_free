import os
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (QAction, QDialog, QFormLayout, QSpinBox,
                                 QDialogButtonBox, QLabel, QCheckBox)
from qgis.core import (QgsApplication, QgsFillSymbol, QgsProject, QgsSettings,
                       QgsVectorLayer, QgsWkbTypes)

from .tree_tool import TreeCaptureTool

LAYER_NAME = 'Tree crowns'
KEY = 'MDTreeFree/'
DEFAULTS = {'radius': 100, 'bias': 0, 'refine': 2, 'nodes': 120, 'edge': 7, 'shadow': 1, 'snap': 1, 'hull': 1}


class SettingsDialog(QDialog):
    def __init__(self, vals, parent=None):
        super().__init__(parent)
        self.setWindowTitle('MD_TreeFree - settings')
        form = QFormLayout(self)
        self.sp = {}
        rows = [('radius', 'Search radius around the click (screen pixels)', 20, 400),
                ('bias', 'Crown size bias (negative = tighter, positive = looser)', -10, 10),
                ('refine', 'Refinement passes (more = better fit, slower)', 0, 4),
                ('nodes', 'Nodes (vertices) per crown outline', 24, 720),
                ('edge', 'Edge smoothness (0 = raw, 10 = very round)', 0, 10)]
        for k, label, lo, hi in rows:
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(vals[k])
            self.sp[k] = w
            form.addRow(label, w)
        self.shadow = QCheckBox('Ignore cast shadows next to the tree')
        self.shadow.setChecked(bool(vals['shadow']))
        form.addRow(self.shadow)
        self.hull = QCheckBox('Freehand look: smooth, full crown without notches')
        self.hull.setChecked(bool(vals['hull']))
        form.addRow(self.hull)
        self.snap = QCheckBox('Snap outline to the crown edge (more accurate, follows leaf detail)')
        self.snap.setChecked(bool(vals['snap']))
        form.addRow(self.snap)
        form.addRow(QLabel(
            'Click near the middle of a tree = automatic outline.\n'
            'Click and drag = draw a circle by hand (centre to edge).\n'
            'Right click = delete the crown under the cursor.\n'
            'Make the search radius a bit bigger than the biggest tree.'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        v = {k: w.value() for k, w in self.sp.items()}
        v['shadow'] = int(self.shadow.isChecked())
        v['snap'] = int(self.snap.isChecked())
        v['hull'] = int(self.hull.isChecked())
        return v


class TreeCrownPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self._layer_id = None

    def initGui(self):
        self.action = QAction(QIcon(os.path.join(os.path.dirname(__file__), 'icon.svg')),
                              'MD_TreeFree', self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.activate)
        self.action_set = QAction('MD_TreeFree settings...', self.iface.mainWindow())
        self.action_set.triggered.connect(self.show_settings)
        self.tool = TreeCaptureTool(self)
        self.tool.setAction(self.action)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('MD_TreeFree', self.action)
        self.iface.addPluginToMenu('MD_TreeFree', self.action_set)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu('MD_TreeFree', self.action)
        self.iface.removePluginMenu('MD_TreeFree', self.action_set)
        if self.canvas.mapTool() is self.tool:
            self.canvas.unsetMapTool(self.tool)

    def activate(self):
        self.canvas.setMapTool(self.tool)

    # ---- settings ----
    def settings(self):
        s = QgsSettings()
        return {k: int(s.value(KEY + k, v)) for k, v in DEFAULTS.items()}

    def show_settings(self):
        dlg = SettingsDialog(self.settings(), self.iface.mainWindow())
        if dlg.exec_() if hasattr(dlg, 'exec_') else dlg.exec():
            s = QgsSettings()
            for k, v in dlg.values().items():
                s.setValue(KEY + k, v)

    # ---- output layer ----
    def target_layer(self, create=True):
        active = self.iface.activeLayer()
        if (isinstance(active, QgsVectorLayer)
                and active.geometryType() == QgsWkbTypes.PolygonGeometry):
            return active
        lyr = QgsProject.instance().mapLayer(self._layer_id) if self._layer_id else None
        if lyr is not None or not create:
            return lyr
        crs = self.canvas.mapSettings().destinationCrs().authid()
        lyr = QgsVectorLayer(f'Polygon?crs={crs}&field=id:integer&field=area_m2:double',
                             LAYER_NAME, 'memory')
        lyr.renderer().setSymbol(QgsFillSymbol.createSimple({
            'color': '0,200,0,90', 'outline_color': '255,0,0,255', 'outline_width': '0.6'}))
        QgsProject.instance().addMapLayer(lyr)
        self._layer_id = lyr.id()
        return lyr
