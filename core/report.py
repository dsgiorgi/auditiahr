
from reportlab.pdfgen import canvas
def export_pdf(summary, path):
    c=canvas.Canvas(path); y=800
    c.drawString(72,y,"Reporte de Auditoría de Fairness")
    for k,v in (summary or {}).items():
        y-=18; c.drawString(72,y,f"{k}: {v}")
    c.save()
