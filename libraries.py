import sys
import includes
from definition import gap
from definition import gapPlot
from definition import gapPlotMeta
from definition import gapPlotMetaData
filename = sys.argv[-1]
if filename == 'agis.gml' or filename == 'Agis.gml':
  gap()
if filename == 'Bestel.gml' or filename == 'bestel.gml':
  gapPlot()
if filename == 'Cwix.graphml' or filename == 'cwix.graphml':
  gapPlotMeta()
if filename == 'Cogentco.gml' or filename == 'cogentco.gml':
  gapPlotMetaData()