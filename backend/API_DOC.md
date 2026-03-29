# Transport Optimizer API

Запуск dev:
```bash
uvicorn api:app --reload --port 8000
```
На старте загружаются: модели LightGBM, фичи из train, map route→office, конфиг `vehicles.json` (парк + штрафы).

## Эндпоинты

### POST /optimize
Расчёт плана по маршруту/времени на горизонты A/B/C/D.

Request:
```json
{
  "route_id": "29",
  "timestamp": "2025-05-30T10:30:00",
  "initial_stock_units": 150,          // опц, иначе из конфига
  "route_distance_km": 15,             // опц, иначе из конфига
  "wait_penalty_per_minute": 8,        // опц, иначе из конфига
  "underload_penalty_per_unit": 5      // опц, иначе из конфига
}
```

Response:
```json
{
  "plan": [
    {
      "office_from_id": "26",
      "route_id": "29",
      "timestamp": "2025-05-30 10:30:00",
      "horizon": "A: now",
      "vehicle_type": "truck_l",
      "vehicles_count": 2,
      "demand": 150.0,
      "covered": 120.0,
      "cost_fixed": 2100.0,
      "cost_underload": 0.0,
      "cost_wait": 0.0,
      "cost_total": 2100.0
    },
    { "horizon": "B: +2h", ... },
    { "horizon": "C: +4h", ... },
    { "horizon": "D: +6h", ... }
  ],
  "coverage_min": 0.0
}
```
- `plan` — строка на каждый тип ТС в каждом горизонте.
- `cost_total = cost_fixed + cost_underload + cost_wait`.
- `coverage_min` — min(covered − demand) по всем слотам (≥ 0, если покрыто).

### GET /config
Возвращает текущий конфиг (в памяти / vehicles.json).

### POST /config
Записывает конфиг целиком (перезапись). Тело — такая же структура, как GET /config.

### GET /vehicles
Список всех типов ТС.
Ответ (пример):
```json
[
  {"vehicle_type":"gazelle_s","capacity_units":18,"cost_per_km":40,"available":5},
  ...
]
```

### POST /vehicles
Добавить тип ТС.
Тело:
```json
{"vehicle_type":"van","capacity_units":22,"cost_per_km":35,"available":3}
```
Ошибка 400, если vehicle_type уже существует.

### PATCH /vehicles/{vehicle_type}
Изменить существующий тип (замена на присланные поля).
Тело как у POST /vehicles. Ошибка 404, если нет такого vehicle_type.

### DELETE /vehicles/{vehicle_type}
Удалить тип. Ошибка 404, если нет такого vehicle_type.

### PATCH /settings
Частичное обновление штрафов/параметров (только указанные поля меняются):
```json
{
  "underload_penalty_per_unit": 6,
  "wait_penalty_per_minute": 9,
  "initial_stock_units": 180,
  "route_distance_km": 18
}
```
Ответ: `{ "status": "ok", "settings": { ...применённые поля... } }`

### GET /health
Возвращает `{ "status": "ok" }`.

## Формат vehicles.json (по умолчанию)
```json
{
  "vehicles": [
    {"vehicle_type":"gazelle_s","capacity_units":18,"cost_per_km":40,"available":5},
    {"vehicle_type":"gazelle_l","capacity_units":26,"cost_per_km":45,"available":4},
    {"vehicle_type":"truck_m","capacity_units":40,"cost_per_km":60,"available":3},
    {"vehicle_type":"truck_l","capacity_units":60,"cost_per_km":70,"available":2}
  ],
  "underload_penalty_per_unit": 5,
  "wait_penalty_per_minute": 8,
  "initial_stock_units": 150,
  "route_distance_km": 15
}
```

## Логика расчёта (кратко)
- Горизонты: A=initial_stock, B=pred_0_2h, C=pred_2_4h, D=pred_4_6h (предикты LightGBM).
- Спрос округляется вверх (ceil).
- Подбор машин: SLSQP минимизирует `fixed_cost + under_penalty * max(covered - demand, 0)`
  при ограничениях covered≥demand и 0≤x_i≤available_i. После округления добивка самым дешёвым, если не хватает покрытия.
- Стоимость рейса: `cost_per_km * route_distance_km`.
- Штраф ожидания: когорты товаров, сумма size*(horizon_min − arrival_min)*wait_penalty (A=0; B: initial*120; C: initial*240 + B*120; D: initial*360 + B*240 + C*120).
- Выход — разложение по типам ТС и по горизонту, плюс coverage_min.

## Роль фронта
- Редактирует парк/штрафы через /vehicles и /settings (или /config целиком).
- Для расчёта вызывает /optimize с route_id, timestamp (+ overrides при желании).
- Отображает план и стоимости из ответа, без пересчётов на клиенте.
