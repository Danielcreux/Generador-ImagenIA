#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Imágenes con IA usando Stability AI
Optimizado con múltiples hilos para máximo rendimiento en CPU multi-núcleo
"""

import os
import sys
import time
import json
import yaml
import threading
import queue
import base64
from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Librerías externas
import requests
from PIL import Image
from colorama import init, Fore, Style
import psutil
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Inicializar colorama (para colores en terminal)
init(autoreset=True)

# Configurar logging - SIN EMOJIS para evitar errores en Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generador_imagenes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EstadoTarea(Enum):
    PENDIENTE = "pendiente"
    PROCESANDO = "procesando"
    COMPLETADA = "completada"
    FALLIDA = "fallida"
    CANCELADA = "cancelada"


@dataclass
class TareaGeneracion:
    id: str
    prompt: str
    estado: EstadoTarea
    timestamp: float
    resultado: Optional[str] = None
    error: Optional[str] = None
    tiempo_procesamiento: Optional[float] = None
    worker_id: Optional[int] = None


class MonitorProgreso:
    def __init__(self, total_tareas: int):
        self.total_tareas = total_tareas
        self.tareas_completadas = 0
        self.tareas_fallidas = 0
        self.lock = threading.Lock()
        self.inicio = time.time()
    
    def tarea_completada(self, exitosa: bool = True):
        with self.lock:
            if exitosa:
                self.tareas_completadas += 1
            else:
                self.tareas_fallidas += 1
            self._mostrar_progreso()
    
    def _mostrar_progreso(self):
        completado = self.tareas_completadas + self.tareas_fallidas
        porcentaje = (completado / self.total_tareas) * 100
        tiempo_transcurrido = time.time() - self.inicio
        
        barra = self._generar_barra_progreso(porcentaje)
        
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.write(
            f"{Fore.CYAN}Progreso: {barra} {Fore.YELLOW}{porcentaje:.1f}% "
            f"{Fore.GREEN}[OK:{self.tareas_completadas}] "
            f"{Fore.RED}[Error:{self.tareas_fallidas}] "
            f"{Fore.WHITE}Tiempo: {tiempo_transcurrido:.1f}s"
        )
        sys.stdout.flush()
    
    def _generar_barra_progreso(self, porcentaje: float, longitud: int = 30) -> str:
        completado = int(longitud * porcentaje / 100)
        barra = f"{Fore.GREEN}{'█' * completado}{Fore.WHITE}{'░' * (longitud - completado)}"
        return barra


class GeneradorImagenesIA:
    """
    Clase principal que maneja la generación de imágenes con Stability AI.
    """
    
    def __init__(self, archivo_config: str = "config.yaml"):
        self.config = self._cargar_config(archivo_config)
        self._inicializar_directorios()
        self._verificar_configuracion()
        
        self.cola_tareas = queue.Queue()
        self.resultados = []
        self.tareas_activas = []
        
        self.detener_procesamiento = threading.Event()
        self.lock_resultados = threading.Lock()
        
        self.estadisticas = {
            'total_tareas': 0,
            'tareas_completadas': 0,
            'tareas_fallidas': 0,
            'tiempo_total': 0,
            'promedio_por_imagen': 0
        }
        
        logger.info(f"Generador inicializado con {self._obtener_info_sistema()}")
    
    def _cargar_config(self, archivo_config: str) -> Dict[str, Any]:
        try:
            with open(archivo_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {archivo_config}")
            return config
        except FileNotFoundError:
            logger.warning("Archivo de configuración no encontrado. Usando valores por defecto.")
            return {
                'max_trabajadores': 4,
                'tamano_lote': 5,
                'resolucion': '1024x1024',
                'directorio_salida': 'imagenes_generadas',
                'archivo_prompts': 'prompts.txt',
                'modo': 'batch',
                'api': {
                    'proveedor': 'stabilityai',
                    'stabilityai': {
                        'api_key_env': 'STABILITY_API_KEY',
                        'model': 'stable-diffusion-xl-1024-v1-0',
                        'output_format': 'png',
                        'cfg_scale': 7,
                        'steps': 30
                    }
                }
            }
    
    def _inicializar_directorios(self):
        directorio_salida = self.config.get('directorio_salida', 'imagenes_generadas')
        Path(directorio_salida).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de salida: {directorio_salida}")
    
    def _verificar_configuracion(self):
        """Verifica que la configuración de Stability AI sea correcta y que la API key exista."""
        proveedor = self.config.get('api', {}).get('proveedor', 'stabilityai')
        if proveedor != 'stabilityai':
            logger.error(f"Proveedor '{proveedor}' no soportado. Debe ser 'stabilityai'.")
            sys.exit(1)
        
        api_key_env = self.config['api']['stabilityai'].get('api_key_env', 'STABILITY_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.error(f"Variable de entorno {api_key_env} no encontrada. "
                         "Crea un archivo .env con tu clave de Stability AI.")
            sys.exit(1)
        
        # Opcional: probar la conexión con Stability AI (list engines)
        try:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.get("https://api.stability.ai/v1/engines/list", headers=headers, timeout=5)
            if response.status_code == 200:
                logger.info("Conexion con Stability AI exitosa")
            else:
                logger.error(f"Error conectando con Stability AI: {response.status_code} - {response.text}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"No se pudo conectar con Stability AI: {e}")
            sys.exit(1)
    
    def _obtener_info_sistema(self) -> str:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        memoria = psutil.virtual_memory()
        return (f"[CPU: {cpu_count} nucleos, "
                f"Frec: {cpu_freq.current:.0f}MHz, "
                f"RAM: {memoria.total / (1024**3):.1f}GB]")
    
    def _generar_nombre_archivo(self, prompt: str, worker_id: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_prompt = hashlib.md5(prompt.encode()).hexdigest()[:8]
        prompt_limpio = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        prompt_limpio = prompt_limpio.replace(' ', '_')
        directorio = self.config.get('directorio_salida', 'imagenes_generadas')
        nombre = f"{directorio}/img_{timestamp}_w{worker_id}_{hash_prompt}_{prompt_limpio}.png"
        return nombre
    
    def _generar_con_stabilityai(self, prompt: str, worker_id: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """Genera una imagen usando la API de Stability AI."""
        api_key_env = self.config['api']['stabilityai'].get('api_key_env', 'STABILITY_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            return False, None, "Clave API de Stability AI no encontrada"
        
        model = self.config['api']['stabilityai'].get('model', 'stable-diffusion-xl-1024-v1-0')
        output_format = self.config['api']['stabilityai'].get('output_format', 'png')
        resolucion = self.config.get('resolucion', '1024x1024')
        # Parsear resolución a ancho y alto
        try:
            ancho, alto = map(int, resolucion.lower().split('x'))
        except:
            ancho, alto = 1024, 1024
        
        cfg_scale = self.config['api']['stabilityai'].get('cfg_scale', 7)
        steps = self.config['api']['stabilityai'].get('steps', 30)
        seed = self.config['api']['stabilityai'].get('seed')
        
        # Endpoint para el modelo
        url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": cfg_scale,
            "height": alto,
            "width": ancho,
            "steps": steps,
            "samples": 1,
            "output_format": output_format
        }
        if seed is not None:
            body["seed"] = seed
        
        for intento in range(3):
            try:
                response = requests.post(url, json=body, headers=headers, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    image_data = base64.b64decode(data["artifacts"][0]["base64"])
                    nombre_archivo = self._generar_nombre_archivo(prompt, worker_id)
                    with open(nombre_archivo, "wb") as f:
                        f.write(image_data)
                    return True, nombre_archivo, None
                elif response.status_code == 429:
                    logger.warning(f"Limite de tasa excedido, reintentando en {2**intento}s...")
                    time.sleep(2 ** intento)
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    return False, None, error_msg
            except Exception as e:
                if intento == 2:
                    return False, None, str(e)
                time.sleep(1)
        return False, None, "Maximo de reintentos alcanzado"
    
    def _generar_imagen(self, prompt: str, worker_id: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """Dispatcher que llama al método según el proveedor configurado."""
        proveedor = self.config.get('api', {}).get('proveedor', 'stabilityai')
        if proveedor == 'stabilityai':
            return self._generar_con_stabilityai(prompt, worker_id)
        else:
            return False, None, f"Proveedor '{proveedor}' no soportado"
    
    def _worker(self, worker_id: int):
        logger.info(f"Worker {worker_id} iniciado")
        while not self.detener_procesamiento.is_set():
            try:
                tarea = self.cola_tareas.get(timeout=1)
            except queue.Empty:
                continue
            
            if tarea is None:
                break
            
            tarea.estado = EstadoTarea.PROCESANDO
            tarea.worker_id = worker_id
            
            inicio = time.time()
            exito, ruta, error = self._generar_imagen(tarea.prompt, worker_id)
            tiempo = time.time() - inicio
            
            with self.lock_resultados:
                if exito:
                    tarea.estado = EstadoTarea.COMPLETADA
                    tarea.resultado = ruta
                    tarea.tiempo_procesamiento = tiempo
                    self.resultados.append({
                        'prompt': tarea.prompt,
                        'archivo': ruta,
                        'tiempo': tiempo,
                        'worker': worker_id
                    })
                    logger.info(f"Worker {worker_id} completo: {os.path.basename(ruta)} en {tiempo:.2f}s")
                else:
                    tarea.estado = EstadoTarea.FALLIDA
                    tarea.error = error
                    logger.error(f"Worker {worker_id} fallo: {error}")
            
            self.cola_tareas.task_done()
        
        logger.info(f"Worker {worker_id} finalizado")
    
    def procesar_prompts_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        num_trabajadores = min(
            self.config.get('max_trabajadores', 4),
            len(prompts)
        )
        
        logger.info(f"Iniciando procesamiento de {len(prompts)} prompts con {num_trabajadores} trabajadores")
        
        # Crear tareas
        for i, prompt in enumerate(prompts):
            tarea = TareaGeneracion(
                id=f"task_{i}_{int(time.time())}",
                prompt=prompt,
                estado=EstadoTarea.PENDIENTE,
                timestamp=time.time()
            )
            self.cola_tareas.put(tarea)
        
        monitor = MonitorProgreso(len(prompts))
        
        # Iniciar workers
        workers = []
        for i in range(num_trabajadores):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            workers.append(worker)
        
        # Esperar a que todas las tareas se completen
        self.cola_tareas.join()
        
        # Detener workers
        self.detener_procesamiento.set()
        for _ in workers:
            self.cola_tareas.put(None)
        
        for worker in workers:
            worker.join(timeout=2)
        
        print()
        
        exitosas = len(self.resultados)
        fallidas = len(prompts) - exitosas
        
        logger.info(f"\n{'='*50}")
        logger.info(f"RESUMEN FINAL")
        logger.info(f"{'='*50}")
        logger.info(f"Imagenes generadas: {exitosas}")
        logger.info(f"Fallos: {fallidas}")
        if exitosas > 0:
            tiempo_promedio = sum(r['tiempo'] for r in self.resultados) / exitosas
            logger.info(f"Tiempo promedio por imagen: {tiempo_promedio:.2f}s")
        
        return self.resultados
    
    def procesar_interactivo(self):
        print(f"\n{Fore.CYAN}MODO INTERACTIVO - Generador de Imagenes con Stability AI")
        print(f"{Fore.WHITE}Escribe 'salir' para terminar, 'stats' para ver estadisticas\n")
        
        prompts = []
        while True:
            try:
                prompt = input(f"{Fore.GREEN}Prompt > {Fore.WHITE}").strip()
                
                if prompt.lower() == 'salir':
                    break
                elif prompt.lower() == 'stats':
                    self._mostrar_estadisticas()
                    continue
                elif not prompt:
                    continue
                
                prompts.append(prompt)
                
                if len(prompts) >= self.config.get('tamano_lote', 5):
                    print(f"\n{Fore.YELLOW}Procesando lote de {len(prompts)} imagenes...")
                    resultados = self.procesar_prompts_batch(prompts)
                    self._mostrar_resultados(resultados)
                    prompts = []
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupcion detectada...")
                if prompts:
                    print(f"Procesando ultimos {len(prompts)} prompts...")
                    resultados = self.procesar_prompts_batch(prompts)
                    self._mostrar_resultados(resultados)
                break
        
        if prompts:
            print(f"\n{Fore.YELLOW}Procesando ultimos {len(prompts)} prompts...")
            resultados = self.procesar_prompts_batch(prompts)
            self._mostrar_resultados(resultados)
    
    def _mostrar_resultados(self, resultados):
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"Resultados del lote:")
        for r in resultados:
            nombre = os.path.basename(r['archivo'])
            print(f"{Fore.GREEN} {nombre} - {r['tiempo']:.2f}s (Worker {r['worker']})")
    
    def _mostrar_estadisticas(self):
        print(f"\n{Fore.CYAN}ESTADISTICAS")
        print(f"Tareas completadas: {self.estadisticas['tareas_completadas']}")
        print(f"Tareas fallidas: {self.estadisticas['tareas_fallidas']}")
        if self.estadisticas['tiempo_total'] > 0:
            print(f"Tiempo total: {self.estadisticas['tiempo_total']:.2f}s")
    
    def cargar_prompts_desde_archivo(self, archivo: str = None) -> List[str]:
        if archivo is None:
            archivo = self.config.get('archivo_prompts', 'prompts.txt')
        
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Cargados {len(prompts)} prompts desde {archivo}")
            return prompts
        except FileNotFoundError:
            logger.error(f"Archivo {archivo} no encontrado")
            return []
    
    def ejecutar(self):
        print(f"\n{Fore.CYAN}{'*'*50}")
        print(f"{Fore.YELLOW}   GENERADOR DE IMAGENES CON STABILITY AI - MODO MULTIHILO")
        print(f"{Fore.CYAN}{'*'*50}\n")
        
        print(f"{Fore.WHITE}Configuracion:")
        print(f"   • Proveedor: Stability AI ({self.config['api']['stabilityai']['model']})")
        print(f"   • Trabajadores: {self.config.get('max_trabajadores')}")
        print(f"   • Resolucion: {self.config.get('resolucion')}")
        print(f"   • Directorio salida: {self.config.get('directorio_salida')}")
        
        modo = self.config.get('modo', 'batch')
        
        if modo == 'interactive':
            self.procesar_interactivo()
        else:
            prompts = self.cargar_prompts_desde_archivo()
            if prompts:
                resultados = self.procesar_prompts_batch(prompts)
                archivo_resultados = f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(archivo_resultados, 'w', encoding='utf-8') as f:
                    json.dump(resultados, f, indent=2, ensure_ascii=False)
                print(f"\n{Fore.GREEN}Resultados guardados en: {archivo_resultados}")
            else:
                logger.error("No hay prompts para procesar. Usando modo interactivo...")
                self.procesar_interactivo()


def main():
    try:
        if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
            print(f"{Fore.YELLOW}No estas en un entorno virtual. Se recomienda usar 'venv'")
            respuesta = input("Continuar de todas formas? (s/n): ")
            if respuesta.lower() != 's':
                sys.exit(0)
        
        generador = GeneradorImagenesIA("config.yaml")
        generador.ejecutar()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW} Programa detenido por el usuario")
    except Exception as e:
        print(f"{Fore.RED}Error inesperado: {e}")
        logger.exception("Error fatal")
        sys.exit(1)


if __name__ == "__main__":
    main()