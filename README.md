# Trading-Bot-DCA

Bot automático de criptomonedas hecho en Python con conexión a la Api de Binance. 
El bot realiza DCA y es totalmente calibrable. 
En este archivo instructivo se encuentran todos los pasos para ponerlo en funcionamiento. 

# Donaciones

USDT (TRC20): TP2DxAk2idhxzrb7mKMjt6gTdBeDjjt1zA

ETH  (ETH)  : 0x5c1cd5d71634383dF70F8c093B543D7B31104fbd


# Instrucciones para iniciar la App:


# Instalar Visual Studio Code:

https://code.visualstudio.com/


# Instalar Node.Js

https://nodejs.org/en

Si Node.Js lanza error en la instalación o demasiados warnings: Abrir nuevamente el instalador de Node.Js y colocar Repair o Reparar.


# Abrir una Terminal en Visual Studio Code y colocar las siguientes sentencias y darles enter:

pip install pandas numpy requests pandas-ta setuptools env 

pip install python-dotenv

pip install matplotlib  


# Buscar un directorio similar en tu pc

C:\Users\Nico\AppData\Roaming\Python\Python312\site-packages\pandas_ta\momentum

Y modificar el archivo: squeeze_pro.py

Donde dice:	    	from numpy import NaN as npNaN

Reemplazar con: 	from numpy import nan as npNaN


# Buscar en Telegram el Bot: @BotFather
 
Iniciar una conversación con BotFather y colocar /newbot 

Luego ponerle nombre a tu App de Telegram

Luego nombredetuApp_bot

Luego GUARDAR EL TOKEN QUE TE GENERO EL BOT DE TELEGRAM.

Luego Iniciar una conversación con tu propio bot.

Luego abrir:

https://api.telegram.org/botTU_TOKEN_TELEGRAM/getUpdates

recuerda que la palabra "bot" va delante de tu token

Luego escribirle a tu bot “HOLA”

Te aparecerá un texto con unos datos si actualizas la pagina que abriste.

Deberás copiar el los números que aparecen luego de este texto: "chat":{"id":

Y copiar tanto el tu TOKEN como el CHAT_ID en tu archivo ve.env. Abrirlo con el Bloc de Notas.


# Crear una API HMAC en Binance y guardar la API KEY y API_SECRET en el archivo ve.env

Asegurarse que la Api tiene permiso para usar Spot.

Antes de lanzar Calibrar el archivo VE.ENV


# A continuación, tienes una explicación detallada de las variables que suelen aparecer en el archivo ve.env del bot:

------------------------------------------
- CONFIGURACIÓN DE CREDENCIALES BINANCE
------------------------------------------
- Clave de la API de Binance.
- Debes obtenerla desde tu cuenta de Binance (API Management).
BINANCE_API_KEY=TU_API_KEY_AQUI

- Secreto de la API de Binance.
- Debes obtenerlo al crear la clave en Binance. 
- Guarda esta información en un lugar seguro.
BINANCE_API_SECRET=TU_API_SECRET_AQUI


------------------------------------------
- CONFIGURACIÓN DE CREDENCIALES TELEGRAM
------------------------------------------
- Token del Bot de Telegram.
- Se obtiene al crear un bot con @BotFather en Telegram.
TELEGRAM_BOT_TOKEN=TU_TELEGRAM_BOT_TOKEN_AQUI

- ID del Chat de Telegram a donde se enviarán las notificaciones.
- Puede ser un chat personal o un canal/grupo. Necesitarás obtener este ID.
TELEGRAM_CHAT_ID=TU_TELEGRAM_CHAT_ID_AQUI


------------------------------------------
- CONFIGURACIÓN DEL SÍMBOLO A OPERAR
------------------------------------------
- Par de trading a utilizar. Ejemplo: BTCUSDT, ETHUSDT, etc.
SIMBOL=BTCUSDT


------------------------------------------
- CONFIGURACIÓN DE ESTRATEGIA DE DCA (Dollar-Cost Averaging)
------------------------------------------
-Porcentaje inicial de fondos en USDT que se usan en la primera compra.
- Ejemplo: 1.0 significa 5% de tu balance USDT.
START_PERCENTAGE=5

- Porcentaje a incrementar por cada DCA adicional.
- Cada vez que el precio baja el porcentaje de USDT para la siguiente compra se incrementa.
INCREMENT_PERCENTAGE=2.5

- Porcentaje máximo de USDT que se usará para una sola operación DCA.
MAX_PERCENTAGE=100


------------------------------------------
- CONFIGURACIÓN DE STOP LOSS
------------------------------------------
- Porcentaje de pérdida en el cual se activa el stop loss.
- Ejemplo: 1.0 significa que si el precio cae un 100% por debajo del precio promedio de compra, se vende.
- Si se confia en el DCA dejar este valor en 100 si se quiere calibrar modificarlo a consciencia.
STOP_LOSS_PERCENT=100


------------------------------------------
- CONFIGURACIÓN DE DCA BASADO EN CAÍDA DE PRECIO
------------------------------------------
- Porcentaje de caída del precio desde la última compra para iniciar DCA.
DCA_PRICE_DROP_START=2

- Incremento de ese porcentaje en cada compra DCA subsecuente.
DCA_PRICE_DROP_INCREMENT=4

- Porcentaje máximo de caída a tener en cuenta para el DCA.
DCA_PRICE_DROP_MAX=100


------------------------------------------
- CONFIGURACIÓN DE BNB PARA FEES
------------------------------------------
- Monto mínimo en BNB (en dólares) que se desea mantener para cubrir comisiones.
BNB_MIN_USD=20.0

- Monto en dólares de BNB a reponer cuando el balance en BNB está por debajo del mínimo.
BNB_REPLENISH_USD=50

- Tiempo mínimo (en segundos) entre compras de BNB.
BNB_COOLDOWN_SECONDS=60

- Límite de compras de BNB por día para controlar la frecuencia de reabastecimiento.
BNB_BUY_LIMIT_PER_DAY=3

------------------------------------------
- OTRAS CONFIGURACIONES
------------------------------------------
- Porcentaje de caída desde el máximo para generar una señal (por ejemplo, 2%).
DROP_PERCENT=2

- Porcentaje mínimo de ganancia al cual se considerará vender.
MIN_PROFIT_PERCENT=2


# En Visual Studio Code compilar el archivo .py

python -m compileall mtb5.py   

Finalmente dobleclick en el archivo mtb5.pyc
