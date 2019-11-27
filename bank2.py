# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:54:14 2019

@author: 91974
"""

import io
from PIL import Image
import pytesseract
from wand.image import Image as wi

pdf = wi(filename= "statement_sample1.PDF", resolution = 300)
pdfImg = pdf.convert('jpeg')

imgBlobs = []

for img in pdfImag.sequence:
    page = wi(image = img)
    imgBlobs.append(page.make_blob('jpeg'))
    
extracted_text = []

for ImgBlob in ImgBlobs:
    im = Image.open(io.BytesIO(imgBlob))
    text = pytesseract.image_to_string(im, lang ='eng')
    extracted_text.append(text)
    
print(extracted_text[0])

import camelot

table =camelot.read_pdf("statement_sample1.pdf")


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from io import StringIO

def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = file(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text


text = pdf_to_text("statement_sample1.pdf")


