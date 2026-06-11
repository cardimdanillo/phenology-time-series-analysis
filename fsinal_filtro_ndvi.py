#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:01:35 2026

@author: danillo
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Caminho do arquivo CSV
caminho = '/media/danillo/HD/HD_EXTERNO/01_ACADEMICO/PPGGAG Mestrado Geoprocessamento/disciplinas/Processamento de Sinais/'
arquivo = caminho+"Parque Nacional de Brasilia_ndvi_modis.csv"

# Ler o CSV
df = pd.read_csv(arquivo)

# Selecionar apenas as colunas desejadas
serie_ndvi = df[['date', 'ndvi']].copy()

# Converter a coluna de data para o formato datetime
serie_ndvi['date'] = pd.to_datetime(serie_ndvi['date'])

serie_ndvi = serie_ndvi.sort_values('date')

# Visualizar o resultado
print(serie_ndvi.head())


plt.figure(figsize=(12, 5))

plt.plot(
    serie_ndvi['date'],
    serie_ndvi['ndvi'],
    linewidth=1
)

plt.xlabel('Data')
plt.ylabel('NDVI')
plt.title('Série temporal de NDVI')

# Mostrar apenas os anos no eixo X
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))        # marca a cada ano
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xticks(rotation=0)
plt.tight_layout()

plt.show()

import numpy as np

# FFT
Y = np.fft.rfft(serie_ndvi['ndvi'])

dt = serie_ndvi['date'].diff().dt.days.mean()

fs = 365.25 / dt

print(f"Intervalo médio = {dt:.1f} dias")
print(f"fs = {fs:.2f} observações/ano")

freqs = np.fft.rfftfreq(len(serie_ndvi['date']),1/fs)

plt.figure(figsize=(8,4))

plt.plot(freqs, np.abs(Y)) 

plt.xlabel('Frequência')
plt.ylabel('Amplitude')
plt.xlim(0, 6)
plt.grid()

plt.show()

B = np.ones(len(freqs))
B[freqs>2]=0
Filt = Y*B
filt = np.fft.irfft(Filt,n=len(serie_ndvi))

plt.figure(figsize=(12,5))
plt.plot(serie_ndvi['date'],filt)
plt.show()

# Série harmônica de Fourier

t = np.arange(len(serie_ndvi['date']))

T = 23          # aproximadamente 23 observações por ano
K = 3           # número de harmônicas

X = np.ones((len(t),1))

y = serie_ndvi['ndvi'].values

for k in range(1, K+1):
    X = np.column_stack([
        X,
        np.cos(2*np.pi*k*t/T),
        np.sin(2*np.pi*k*t/T)
    ])
    
coef, *_ = np.linalg.lstsq(X, y, rcond=None)

y_fourier = X @ coef

plt.figure(figsize=(12,5))
plt.plot(serie_ndvi['date'],y_fourier)
plt.show()

# comparar
plt.figure(figsize=(12,5))

plt.plot(
    serie_ndvi['date'],
    y,
    alpha=0.4,
    label='Original'
)

plt.plot(
    serie_ndvi['date'],
    filt,
    linewidth=2,
    label='FFT'
)

plt.plot(
    serie_ndvi['date'],
    y_fourier,
    linewidth=2,
    label='Fourier harmônico'
)

plt.legend()

plt.xlabel('Ano')
plt.ylabel('NDVI')

plt.show()


