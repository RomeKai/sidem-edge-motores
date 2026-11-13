# Simulación – Detección de desgaste en motores con Edge Computing

Trabajo final de **Sistemas en la Industria 4.0 (SI40)** – UTN FRLP.
Romeo Monfroglio (31143) y Emiliano Di Grappa (32298).

## Qué queríamos probar

El profesor está armando un dispositivo de telemetría para motores con un ESP32 y un sensor PZEM-004T. Nuestra
propuesta es que ese ESP32 funcione como **nodo Edge**: que analice los datos ahí mismo, sin depender de la nube, y
que pueda **parar el motor** si detecta un desgaste abrupto.

Como todavía no tenemos el dispositivo funcionando, armamos esta simulación para ver si la idea se sostiene antes de
llevarla al hardware.

## Qué hicimos

**1. Simulamos el sensor.** Generamos datos como los que entrega el PZEM-004T (tensión, corriente, potencia, factor
de potencia y frecuencia) en tres situaciones: motor sano, desgaste gradual y desgaste abrupto. El PZEM no mide
vibración, así que el desgaste lo vemos por su efecto eléctrico: la corriente sube y el factor de potencia baja.

**2. Separamos el nodo Edge en dos caminos:**
- **Tiempo real duro (parada):** si la corriente pasa un umbral crítico dos muestras seguidas, se ordena la parada.
  Es una lógica simple, siempre tarda lo mismo, y por eso se puede garantizar que cumple el plazo (3 muestras).
- **Tiempo real blando (alertas):** aprende cómo se comporta el motor sano y avisa cuando se aleja de eso. Estas
  alertas van al tablero de monitoreo, no paran el motor.

**3. Lo probamos con datos propios.** El camino duro paró el motor a tiempo en las 50 fallas abruptas y el detector
encontró todos los desgastes graduales sin falsas alarmas. Pero esos datos los generamos nosotros, así que era
esperable que funcionara bien.

**4. Lo contrastamos con datos externos.** Usamos el dataset *Azure Predictive Maintenance* (100 máquinas durante un
año, con fallas registradas). Ahí el detector encontró 7 de cada 10 fallas con más de un día de anticipación, pero
**84 de cada 100 alertas fueron falsas**. Además comprobamos que usar más máquinas no mejora el resultado: solo hace
más precisa la medición.

**5. Calculamos cuánto cuesta cada tipo de mantenimiento.** Simulamos 200 motores durante 5 años con mantenimiento
correctivo, preventivo, predictivo y prescriptivo. Con pocas falsas alarmas, el predictivo y el prescriptivo ahorran
mucho. Pero con la tasa de falsas alarmas que medimos, **el predictivo sale más caro que el correctivo**.

**6. Probamos un modelo supervisado.** Entrenamos un modelo que, además de los sensores, usa el registro de errores
y el historial de mantenimiento de cada máquina. Con eso las falsas alarmas bajan a unas 4 por motor por año, lo que
vuelve rentable al mantenimiento predictivo. Aclaración: parte de esa mejora se debe a un patrón propio del dataset
(los mantenimientos caen cada 15 días), así que el resultado más confiable es el que usa sensores y errores.

## Qué proponemos

- **La parada se decide en el borde, con lógica simple y garantizada.** Nunca la decide el detector de anomalías ni
  la nube, porque se equivocan demasiado seguido como para frenar un motor.
- **El ESP32 da la primera alerta** y el **gemelo digital** en la nube la confirma, cruzándola con el contexto de la
  máquina (errores, mantenimientos). Esa vuelta de información hacia la planta, tanto la parada como las acciones
  recomendadas, es lo que convierte el sistema en un gemelo digital y no en un simple tablero.
- **El mantenimiento prescriptivo es lo que hace viable al predictivo:** filtra falsas alarmas y elige el mejor
  momento para intervenir.

## Limitaciones

- Los datos del motor son simulados. Con el sensor real habrá más ruido, y los cambios de carga se pueden confundir
  con desgaste.
- Los tiempos los medimos en una PC con Windows, no en el ESP32. En el dispositivo, el camino duro tiene que ir en una
  tarea de alta prioridad de FreeRTOS.
- El PZEM actualiza más o menos una vez por segundo, así que el sistema no puede reaccionar más rápido que eso.
- Los costos del modelo de mantenimiento son supuestos: sirven para comparar estrategias, no como valores reales.

## Próximos pasos

1. Pasar los dos caminos al firmware del ESP32 y medir los tiempos reales.
2. Tomar datos de un motor real con el dispositivo del profesor y recalibrar los umbrales.
3. Enviar las alertas por MQTT a un tablero de monitoreo.
4. Integrarlo con el gemelo digital que está desarrollando el profesor.

## Cómo correrlo

Requiere Python 3.11 o superior.

```bash
pip install numpy pandas matplotlib scikit-learn pytest
python download_data.py     # baja el dataset de Azure a data/ (~80 MB, una sola vez)
pytest -q                    # pruebas
python run_simulation.py     # simulación, gráficos y modelo de costos
python validate_azure.py     # contraste con Azure PdM (CSV en data/)
python compare_models.py     # comparación con el modelo supervisado
python dashboard.py          # tablero con todas las métricas (se abre en el navegador)
```

### El tablero

`python dashboard.py` levanta un servidor local (solo en tu PC, en `http://127.0.0.1:8765`) y abre el
tablero. Arriba tiene botones para correr cada script: **Motor simulado**, **Contraste con Azure**,
**Modelo supervisado**, **Correr todo** y **Pruebas**. Cuando un script termina, las métricas y los gráficos
se actualizan solos. Si cambiamos datos o parámetros, basta con volver a apretar el botón.

Hace falta el servidor porque un navegador, por seguridad, no puede ejecutar programas de la PC. El
servidor solo acepta esos cinco botones, no cualquier comando. Se cierra con `Ctrl+C` en la terminal.

Los gráficos se amplían con un clic, y otro clic sobre la imagen la muestra en tamaño real.

También queda una copia estática en `results/dashboard.html` (sin botones). Funciona sin internet, así que se
puede mostrar en clase abriéndola directamente con el navegador. Para generarla sin abrir el servidor:
`python dashboard.py --build`.

> En Windows, si la carpeta tiene una ruta muy larga, conviene crear el entorno virtual en una ruta corta
> (por ejemplo `C:\venv`), porque NumPy y Matplotlib pueden fallar al cargar sus librerías.

Los gráficos y tablas quedan en `results/`. Los datos de Azure se descargan del repositorio oficial de Microsoft
(*SQL-Server-R-Services-Samples / PredictiveMaintenanceModelingGuide*), con las 100 máquinas del dataset. No
los subimos al repositorio porque pesan unos 80 MB y son de Microsoft: `download_data.py` los baja de la fuente.

## Organización

| Carpeta / archivo | Contenido |
| --- | --- |
| `si40sim/` | Código de la simulación: sensor, nodo Edge, mantenimiento y modelos |
| `tests/` | Pruebas automáticas |
| `run_simulation.py` | Simulación con datos propios y modelo de costos |
| `validate_azure.py` | Prueba del detector con el dataset de Azure |
| `compare_models.py` | Detector simple vs. modelo supervisado |
| `dashboard.py` | Tablero con las métricas y botones para correr los scripts |
| `make_architecture_fig.py` | Diagrama de arquitectura del informe |
| `download_data.py` | Descarga el dataset de Azure |
| `data/` | Dataset de Azure (100 máquinas, no se sube al repo) |
| `results/` | Gráficos y tablas generados |
