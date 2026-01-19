# Multi-Agent Reinforcement Learning con TorchRL y VMAS

Este proyecto implementa un entrenamiento multi-agente usando **PPO** con el simulador **VMAS** y la librería **TorchRL**.

---

## Características

- Entrenamiento de múltiples agentes en entornos de navegación (`navigation`) de VMAS.
- Implementación de **política probabilística** con `TanhNormal`.
- Uso de **TensorDictModule** para organizar observaciones y acciones por agente.
- Recolección de datos con `SyncDataCollector` y almacenamiento en `ReplayBuffer`.
- Estimación de ventajas usando **GAE (Generalized Advantage Estimation)**.
- Visualización de recompensa media por iteración.
- Renderizado opcional de episodios en GIF (headless usando `pyvirtualdisplay`).

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt


🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
