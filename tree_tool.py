import numpy as np

from qgis.PyQt.QtCore import Qt, QSize
from qgis.PyQt.QtGui import QImage, QColor
from qgis.core import (
    QgsCoordinateTransform, QgsDistanceArea, QgsFeature, QgsFeatureRequest,
    QgsGeometry, QgsMapRendererSequentialJob, QgsMapSettings, QgsPointXY,
    QgsProject, QgsRasterLayer, QgsRectangle, QgsUnitTypes, QgsWkbTypes,
)
from qgis.gui import QgsMapTool, QgsRubberBand

from .segmentation import segment

DRAG_PIXELS = 6  # mouse movement above this = "draw a circle" instead of auto-capture


def qimage_to_rgb(img):
    img = img.convertToFormat(QImage.Format_RGB32)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    n = img.bytesPerLine() * h
    try:
        ptr.setsize(n)
    except AttributeError:
        pass
    arr = np.frombuffer(ptr, np.uint8, count=n).reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
    return arr[..., [2, 1, 0]].copy()  # BGRA (little endian) -> RGB


class TreeCaptureTool(QgsMapTool):
    def __init__(self, plugin):
        super().__init__(plugin.iface.mapCanvas())
        self.plugin = plugin
        self.iface = plugin.iface
        self.canvas = plugin.iface.mapCanvas()
        self.setCursor(Qt.CrossCursor)
        self._press = None
        self._dragging = False
        self.rb = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.rb.setColor(QColor(255, 0, 0, 255))
        self.rb.setFillColor(QColor(0, 200, 0, 70))
        self.rb.setWidth(2)

    def deactivate(self):
        self.rb.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()

    # ---------- helpers ----------
    def _source_raster(self):
        lyr = self.iface.activeLayer()
        if isinstance(lyr, QgsRasterLayer):
            return lyr
        for lay in self.canvas.layers():
            if isinstance(lay, QgsRasterLayer):
                return lay
        return None

    def _render_window(self, raster, pt, radius_px):
        mupp = self.canvas.mapUnitsPerPixel()
        half = radius_px * mupp
        ext = QgsRectangle(pt.x() - half, pt.y() - half, pt.x() + half, pt.y() + half)
        ms = QgsMapSettings()
        ms.setLayers([raster])
        ms.setDestinationCrs(self.canvas.mapSettings().destinationCrs())
        ms.setExtent(ext)
        ms.setOutputSize(QSize(radius_px * 2, radius_px * 2))
        ms.setBackgroundColor(QColor(0, 0, 0))
        job = QgsMapRendererSequentialJob(ms)
        job.start()
        job.waitForFinished()
        return qimage_to_rgb(job.renderedImage()), ext, mupp

    # ---------- events ----------
    def canvasPressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.pos()
            self._dragging = False

    def canvasMoveEvent(self, e):
        if self._press is None or not (e.buttons() & Qt.LeftButton):
            return
        dx = e.pos().x() - self._press.x()
        dy = e.pos().y() - self._press.y()
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > DRAG_PIXELS:
            self._dragging = True
            self.rb.setToGeometry(self._circle(self._press, dist), None)

    def canvasReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self._delete_at(self.toMapCoordinates(e.pos()))
        elif e.button() == Qt.LeftButton and self._press is not None:
            if self._dragging:
                dx = e.pos().x() - self._press.x()
                dy = e.pos().y() - self._press.y()
                self._add_geometry(self._circle(self._press, (dx * dx + dy * dy) ** 0.5))
            else:
                self._capture(self.toMapCoordinates(self._press))
        self.rb.reset(QgsWkbTypes.PolygonGeometry)
        self._press = None
        self._dragging = False

    def _circle(self, center_px, radius_px):
        c = self.toMapCoordinates(center_px)
        return QgsGeometry.fromPointXY(QgsPointXY(c)).buffer(
            radius_px * self.canvas.mapUnitsPerPixel(),
            max(8, self.plugin.settings()['nodes'] // 4))

    # ---------- actions ----------
    def _capture(self, pt):
        raster = self._source_raster()
        if raster is None:
            self.iface.messageBar().pushWarning('MD_TreeFree', 'No raster/imagery layer found.')
            return
        s = self.plugin.settings()
        rgb, ext, mupp = self._render_window(raster, pt, s['radius'])
        pts = segment(rgb, bias=s['bias'] * 0.1, refine=s['refine'],
                      n_ang=s['nodes'], shadow_aware=bool(s['shadow']),
                      snap=bool(s['snap']), smooth_level=s['edge'],
                      hull=bool(s['hull']), grow=0.98 if s['hull'] else 1.0)
        if pts is None:
            self.iface.messageBar().pushInfo(
                'MD_TreeFree',
                'No crown found here. Click nearer the tree centre, zoom in, or drag to draw a circle.')
            return
        ring = [QgsPointXY(ext.xMinimum() + x * mupp, ext.yMaximum() - y * mupp) for x, y in pts]
        ring.append(ring[0])
        geom = QgsGeometry.fromPolygonXY([ring])
        if not geom.isGeosValid():
            geom = geom.makeValid()
        self._add_geometry(geom)

    def _add_geometry(self, geom):
        """geom is in the canvas CRS."""
        if geom is None or geom.isEmpty():
            return
        layer = self.plugin.target_layer()
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        if canvas_crs != layer.crs():
            geom.transform(QgsCoordinateTransform(canvas_crs, layer.crs(), QgsProject.instance()))

        da = QgsDistanceArea()
        da.setSourceCrs(layer.crs(), QgsProject.instance().transformContext())
        da.setEllipsoid(QgsProject.instance().ellipsoid())
        area = da.convertAreaMeasurement(da.measureArea(geom), QgsUnitTypes.AreaSquareMeters)

        if not layer.isEditable():
            layer.startEditing()
        feat = QgsFeature(layer.fields())
        feat.setGeometry(geom)
        idx = layer.fields().indexOf('area_m2')
        if idx >= 0:
            feat.setAttribute(idx, round(area, 2))
        layer.beginEditCommand('Add tree crown')
        layer.addFeature(feat)
        layer.endEditCommand()
        layer.triggerRepaint()

    def _delete_at(self, pt):
        layer = self.plugin.target_layer(create=False)
        if layer is None:
            return
        if not layer.isEditable():
            layer.startEditing()
        src = self.canvas.mapSettings().destinationCrs()
        p = QgsGeometry.fromPointXY(QgsPointXY(pt))
        if src != layer.crs():
            p.transform(QgsCoordinateTransform(src, layer.crs(), QgsProject.instance()))
        rect = p.buffer(self.canvas.mapUnitsPerPixel(), 4).boundingBox()
        for f in layer.getFeatures(QgsFeatureRequest().setFilterRect(rect)):
            if f.geometry().contains(p):
                layer.beginEditCommand('Delete tree crown')
                layer.deleteFeature(f.id())
                layer.endEditCommand()
                layer.triggerRepaint()
                return
