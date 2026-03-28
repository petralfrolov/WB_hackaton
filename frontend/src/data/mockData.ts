import type {
  Warehouse,
  ForecastPoint,
  SankeyData,
  VehicleType,
  TransportRecommendation,
  CostScenario,
  CostBreakdown,
  RiskSettings,
  RouteDistance,
} from '../types'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeForecast(base: number, hours = 8, startHour = 14): ForecastPoint[] {
  return Array.from({ length: hours }, (_, i) => {
    const noise = Math.sin(i * 0.9) * base * 0.08 + (Math.random() - 0.5) * base * 0.05
    const val = Math.round(base + i * base * 0.06 + noise)
    const spread = Math.round(val * 0.12)
    const h = (startHour + i) % 24
    return {
      time: `${String(h).padStart(2, '0')}:00`,
      value: val,
      lower: val - spread,
      upper: val + spread,
    }
  })
}

function makeSankey(
  accepted: number,
  bottleneckAt?: 'sorting' | 'packing' | 'quality' | 'shelf' | 'shipping' | 'waiting' | 'ready',
): SankeyData {
  const nodes: SankeyData['nodes'] = [
    { id: 'accepted', label: 'Принят' },
    { id: 'sorting', label: 'Сортировка' },
    { id: 'packing', label: 'Упаковка' },
    { id: 'quality', label: 'Контроль' },
    { id: 'shelf', label: 'Стеллаж' },
    { id: 'shipping', label: 'Зона отгрузки' },
    { id: 'waiting', label: 'Ожидание ТС' },
    { id: 'ready', label: 'Готов к отгрузке' },
  ]

  type SrcId = 'accepted' | 'sorting' | 'packing' | 'quality' | 'shelf' | 'shipping' | 'waiting'
  type TgtId = 'sorting' | 'packing' | 'quality' | 'shelf' | 'shipping' | 'waiting' | 'ready'

  const seq: Array<[SrcId, TgtId]> = [
    ['accepted', 'sorting'],
    ['sorting', 'packing'],
    ['packing', 'quality'],
    ['quality', 'shelf'],
    ['shelf', 'shipping'],
    ['shipping', 'waiting'],
    ['waiting', 'ready'],
  ]

  let cur = accepted
  const links = seq.map(([src, tgt]) => {
    const isBottleneck = src === bottleneckAt
    const ratio = isBottleneck ? 0.62 : 0.97 - Math.random() * 0.04
    const next = Math.round(cur * ratio)
    const link = { source: src, target: tgt, value: next }
    cur = next
    return link
  })

  return { nodes, links }
}

const COMMON_VEHICLES: VehicleType[] = [
  {
    id: 'v-gazel', name: 'Газель', capacity: 50, costPerKm: 28, available: 8,
    incoming: [{ id: 'inc-gz-1', name: 'Газель', arrivalMinutes: 25, count: 2 }],
  },
  {
    id: 'v-largus', name: 'Ларгус', capacity: 20, costPerKm: 18, available: 5,
    incoming: [{ id: 'inc-lr-1', name: 'Ларгус', arrivalMinutes: 55, count: 1 }],
  },
  {
    id: 'v-fura', name: 'Фура', capacity: 500, costPerKm: 55, available: 3,
    incoming: [],
  },
  {
    id: 'v-micro', name: 'Микроавтобус', capacity: 30, costPerKm: 22, available: 4,
    incoming: [{ id: 'inc-mb-1', name: 'Микроавтобус', arrivalMinutes: 40, count: 1 }],
  },
]

const LIMITED_VEHICLES: VehicleType[] = [
  {
    id: 'v-gazel', name: 'Газель', capacity: 50, costPerKm: 28, available: 4,
    incoming: [{ id: 'inc-gz-2', name: 'Газель', arrivalMinutes: 30, count: 1 }],
  },
  {
    id: 'v-fura', name: 'Фура', capacity: 500, costPerKm: 55, available: 2,
    incoming: [],
  },
]

const SOUTH_VEHICLES: VehicleType[] = [
  {
    id: 'v-gazel', name: 'Газель', capacity: 50, costPerKm: 28, available: 6,
    incoming: [{ id: 'inc-gz-3', name: 'Газель', arrivalMinutes: 45, count: 2 }],
  },
  {
    id: 'v-largus', name: 'Ларгус', capacity: 20, costPerKm: 18, available: 7,
    incoming: [],
  },
  {
    id: 'v-ref', name: 'Рефрижератор', capacity: 200, costPerKm: 48, available: 2,
    incoming: [{ id: 'inc-ref-1', name: 'Рефрижератор', arrivalMinutes: 80, count: 1 }],
  },
]

// ─── Warehouses ───────────────────────────────────────────────────────────────

export const warehouses: Warehouse[] = [
  {
    id: 'w01', name: 'Москва-Восток', city: 'Москва',
    lat: 55.75, lng: 37.62, status: 'ok', readyToShip: 234,
    forecast: makeForecast(230), sankeyData: makeSankey(1200), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w02', name: 'Подольск-1', city: 'Подольск',
    lat: 55.43, lng: 37.54, status: 'warning', readyToShip: 567,
    forecast: makeForecast(550, 8, 13), sankeyData: makeSankey(1800, 'shelf'), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w03', name: 'Химки-Центр', city: 'Химки',
    lat: 55.89, lng: 37.43, status: 'critical', readyToShip: 891,
    forecast: makeForecast(880, 8, 12), sankeyData: makeSankey(2400, 'shipping'),
    vehicles: [
      { id: 'v-gazel', name: 'Газель', capacity: 50, costPerKm: 28, available: 12, incoming: [{ id: 'inc-gz-c', name: 'Газель', arrivalMinutes: 15, count: 3 }] },
      { id: 'v-largus', name: 'Ларгус', capacity: 20, costPerKm: 18, available: 6, incoming: [] },
      { id: 'v-fura', name: 'Фура', capacity: 500, costPerKm: 55, available: 4, incoming: [{ id: 'inc-fu-c', name: 'Фура', arrivalMinutes: 60, count: 1 }] },
    ],
  },
  {
    id: 'w04', name: 'Северная-1', city: 'Санкт-Петербург',
    lat: 59.93, lng: 30.32, status: 'ok', readyToShip: 312,
    forecast: makeForecast(310), sankeyData: makeSankey(1500), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w05', name: 'Казань-Логистик', city: 'Казань',
    lat: 55.79, lng: 49.12, status: 'warning', readyToShip: 445,
    forecast: makeForecast(440, 8, 15), sankeyData: makeSankey(1600, 'ready'), vehicles: LIMITED_VEHICLES,
  },
  {
    id: 'w06', name: 'НН-Южный', city: 'Нижний Новгород',
    lat: 56.32, lng: 44.0, status: 'ok', readyToShip: 198,
    forecast: makeForecast(200), sankeyData: makeSankey(900), vehicles: LIMITED_VEHICLES,
  },
  {
    id: 'w07', name: 'Урал-Хаб', city: 'Екатеринбург',
    lat: 56.83, lng: 60.6, status: 'ok', readyToShip: 523,
    forecast: makeForecast(520), sankeyData: makeSankey(2000), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w08', name: 'Самара-1', city: 'Самара',
    lat: 53.19, lng: 50.15, status: 'warning', readyToShip: 334,
    forecast: makeForecast(330, 8, 13), sankeyData: makeSankey(1400, 'quality'), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w09', name: 'Ростов-Юг', city: 'Ростов-на-Дону',
    lat: 47.23, lng: 39.72, status: 'ok', readyToShip: 276,
    forecast: makeForecast(275), sankeyData: makeSankey(1100), vehicles: SOUTH_VEHICLES,
  },
  {
    id: 'w10', name: 'Краснодар-1', city: 'Краснодар',
    lat: 45.04, lng: 38.98, status: 'ok', readyToShip: 189,
    forecast: makeForecast(185), sankeyData: makeSankey(800), vehicles: SOUTH_VEHICLES,
  },
  {
    id: 'w11', name: 'Воронеж-Центральный', city: 'Воронеж',
    lat: 51.67, lng: 39.2, status: 'ok', readyToShip: 421,
    forecast: makeForecast(420), sankeyData: makeSankey(1700), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w12', name: 'Ярославль-1', city: 'Ярославль',
    lat: 57.62, lng: 39.89, status: 'warning', readyToShip: 388,
    forecast: makeForecast(385, 8, 11), sankeyData: makeSankey(1500, 'packing'), vehicles: LIMITED_VEHICLES,
  },
  {
    id: 'w13', name: 'Тула-Логистик', city: 'Тула',
    lat: 54.19, lng: 37.62, status: 'ok', readyToShip: 156,
    forecast: makeForecast(155), sankeyData: makeSankey(700), vehicles: LIMITED_VEHICLES,
  },
  {
    id: 'w14', name: 'Саратов-1', city: 'Саратов',
    lat: 51.53, lng: 46.03, status: 'ok', readyToShip: 267,
    forecast: makeForecast(265), sankeyData: makeSankey(1050), vehicles: COMMON_VEHICLES,
  },
  {
    id: 'w15', name: 'Уфа-Восток', city: 'Уфа',
    lat: 54.73, lng: 55.96, status: 'ok', readyToShip: 345,
    forecast: makeForecast(340), sankeyData: makeSankey(1350), vehicles: COMMON_VEHICLES,
  },
]

// ─── Route Distances ──────────────────────────────────────────────────────────

export const routeDistances: RouteDistance[] = [
  { id: 'rd01', fromId: 'w03', toId: 'w01', fromCity: 'Химки', toCity: 'Москва-Восток', distanceKm: 42 },
  { id: 'rd02', fromId: 'w02', toId: 'w01', fromCity: 'Подольск', toCity: 'Москва-Восток', distanceKm: 38 },
  { id: 'rd03', fromId: 'w13', toId: 'w01', fromCity: 'Тула', toCity: 'Москва-Восток', distanceKm: 193 },
  { id: 'rd04', fromId: 'w12', toId: 'w01', fromCity: 'Ярославль', toCity: 'Москва-Восток', distanceKm: 265 },
  { id: 'rd05', fromId: 'w04', toId: 'w01', fromCity: 'Санкт-Петербург', toCity: 'Москва-Восток', distanceKm: 714 },
  { id: 'rd06', fromId: 'w11', toId: 'w01', fromCity: 'Воронеж', toCity: 'Москва-Восток', distanceKm: 520 },
  { id: 'rd07', fromId: 'w05', toId: 'w06', fromCity: 'Казань', toCity: 'Нижний Новгород', distanceKm: 395 },
  { id: 'rd08', fromId: 'w08', toId: 'w07', fromCity: 'Самара', toCity: 'Екатеринбург', distanceKm: 655 },
  { id: 'rd09', fromId: 'w09', toId: 'w10', fromCity: 'Ростов-на-Дону', toCity: 'Краснодар', distanceKm: 218 },
  { id: 'rd10', fromId: 'w14', toId: 'w11', fromCity: 'Саратов', toCity: 'Воронеж', distanceKm: 410 },
  { id: 'rd11', fromId: 'w15', toId: 'w07', fromCity: 'Уфа', toCity: 'Екатеринбург', distanceKm: 340 },
  { id: 'rd12', fromId: 'w06', toId: 'w01', fromCity: 'Нижний Новгород', toCity: 'Москва-Восток', distanceKm: 410 },
]

// ─── Cost breakdown helpers ───────────────────────────────────────────────────

function makeBreakdown(
  vehicles: Array<{ name: string; count: number; fixedCost: number; capacity: number; load: number }>,
  wEcon = 1.0,
  wUrg = 1.2,
  avgWait = 45,
  items = 120,
): CostBreakdown {
  return {
    vehicles,
    wEcon,
    wUrg,
    pEmpty: 12,
    pDelay: 8,
    avgWaitMinutes: avgWait,
    itemsWaiting: items,
  }
}

export function computeCostFromBreakdown(b: CostBreakdown): number {
  const fixed = b.vehicles.reduce((s, v) => s + v.fixedCost * v.count, 0)
  const empty = b.wEcon * b.vehicles.reduce((s, v) => s + (v.capacity - v.load) * v.count, 0) * b.pEmpty
  const delay = b.wUrg * b.itemsWaiting * b.avgWaitMinutes * b.pDelay
  return Math.round(fixed + empty + delay)
}

// ─── Optimizer Recommendations ────────────────────────────────────────────────

export const recommendations: TransportRecommendation[] = [
  { id: 'r01', route: 'Химки → Москва-СЦ', warehouseId: 'w03', forecast: 864, recommendation: '2× Фура, 3× Газель', status: 'warning' },
  { id: 'r02', route: 'Подольск → Москва-ЮЖ', warehouseId: 'w02', forecast: 540, recommendation: '3× Газель, 1× Ларгус', status: 'pending' },
  { id: 'r03', route: 'Казань → Казань-СЦ', warehouseId: 'w05', forecast: 428, recommendation: '2× Газель, 2× Ларгус', status: 'pending' },
  { id: 'r04', route: 'Ярославль → Москва-СЦ', warehouseId: 'w12', forecast: 372, recommendation: '2× Газель, 1× Микроавтобус', status: 'pending' },
  { id: 'r05', route: 'Самара → Самара-СЦ', warehouseId: 'w08', forecast: 318, recommendation: '1× Фура, 1× Газель', status: 'called' },
  { id: 'r06', route: 'Воронеж → Москва-ЮЖ', warehouseId: 'w11', forecast: 405, recommendation: '3× Газель', status: 'called' },
  { id: 'r07', route: 'Москва-Восток → МСК-СЦ', warehouseId: 'w01', forecast: 220, recommendation: '1× Газель, 2× Ларгус', status: 'pending' },
  { id: 'r08', route: 'СПб → СПб-Центр', warehouseId: 'w04', forecast: 297, recommendation: '2× Газель', status: 'called' },
  { id: 'r09', route: 'Ростов → Ростов-СЦ', warehouseId: 'w09', forecast: 264, recommendation: '1× Фура', status: 'pending' },
  { id: 'r10', route: 'Уфа → Уфа-Логистик', warehouseId: 'w15', forecast: 331, recommendation: '2× Газель, 1× Ларгус', status: 'pending' },
]

// ─── Cost-Benefit Scenarios ────────────────────────────────────────────────────

export const costScenariosMap: Record<string, CostScenario[]> = {
  r01: [
    {
      id: 'a', name: 'Вариант А', description: '2× Фура + 3× Газель, +20 мин ожидания', cost: 47200, time: '+20 мин',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 2, fixedCost: 8500, capacity: 500, load: 420 },
        { name: 'Газель', count: 3, fixedCost: 2800, capacity: 50, load: 45 },
      ], 0.8, 1.4, 20, 864),
    },
    {
      id: 'b', name: 'Вариант Б', description: '3× Фура, немедленно', cost: 54000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 3, fixedCost: 8500, capacity: 500, load: 288 },
      ], 0.8, 0, 0, 864),
    },
    {
      id: 'c', name: 'Вариант В', description: '5× Фура (с запасом), немедленно', cost: 72500, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 5, fixedCost: 8500, capacity: 500, load: 172 },
      ], 0.8, 0, 0, 864),
    },
  ],
  r02: [
    {
      id: 'a', name: 'Вариант А', description: '3× Газель + 1× Ларгус, +40 мин', cost: 11200, time: '+40 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 3, fixedCost: 2800, capacity: 50, load: 46 },
        { name: 'Ларгус', count: 1, fixedCost: 1800, capacity: 20, load: 18 },
      ], 0.8, 1.2, 40, 540),
    },
    {
      id: 'b', name: 'Вариант Б', description: '2× Газель + 1× Микроавтобус, +15 мин', cost: 13500, time: '+15 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 48 },
        { name: 'Микроавтобус', count: 1, fixedCost: 2200, capacity: 30, load: 25 },
      ], 0.8, 1.2, 15, 540),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Фура, немедленно', cost: 18000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 540 },
      ], 0.8, 0, 0, 540),
    },
  ],
  r03: [
    {
      id: 'a', name: 'Вариант А', description: '2× Газель + 2× Ларгус, +30 мин', cost: 9800, time: '+30 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 45 },
        { name: 'Ларгус', count: 2, fixedCost: 1800, capacity: 20, load: 17 },
      ], 0.8, 1.2, 30, 428),
    },
    {
      id: 'b', name: 'Вариант Б', description: '1× Фура, немедленно', cost: 16500, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 428 },
      ], 0.8, 0, 0, 428),
    },
    {
      id: 'c', name: 'Вариант В', description: '3× Газель, немедленно (повышенный тариф)', cost: 19200, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Газель (экспресс)', count: 3, fixedCost: 3800, capacity: 50, load: 42 },
      ], 0.8, 0, 0, 428),
    },
  ],
  r04: [
    {
      id: 'a', name: 'Вариант А', description: '2× Газель + 1× Микроавтобус, +45 мин', cost: 8600, time: '+45 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 44 },
        { name: 'Микроавтобус', count: 1, fixedCost: 2200, capacity: 30, load: 22 },
      ], 0.8, 1.2, 45, 372),
    },
    {
      id: 'b', name: 'Вариант Б', description: '2× Газель, немедленно', cost: 11000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 47 },
      ], 0.8, 0, 0, 372),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Фура, немедленно', cost: 17200, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 372 },
      ], 0.8, 0, 0, 372),
    },
  ],
  r07: [
    {
      id: 'a', name: 'Вариант А', description: '1× Газель + 2× Ларгус, +20 мин', cost: 7400, time: '+20 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 1, fixedCost: 2800, capacity: 50, load: 44 },
        { name: 'Ларгус', count: 2, fixedCost: 1800, capacity: 20, load: 18 },
      ], 0.8, 1.2, 20, 220),
    },
    {
      id: 'b', name: 'Вариант Б', description: '3× Ларгус, немедленно', cost: 9200, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Ларгус', count: 3, fixedCost: 1800, capacity: 20, load: 15 },
      ], 0.8, 0, 0, 220),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Фура, немедленно', cost: 14800, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 220 },
      ], 0.8, 0, 0, 220),
    },
  ],
  r09: [
    {
      id: 'a', name: 'Вариант А', description: '1× Фура, +35 мин (заполнение)', cost: 15600, time: '+35 мин',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 460 },
      ], 0.8, 1.2, 35, 264),
    },
    {
      id: 'b', name: 'Вариант Б', description: '1× Фура, немедленно', cost: 18000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 264 },
      ], 0.8, 0, 0, 264),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Реф-фура, немедленно', cost: 24000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Рефрижератор', count: 1, fixedCost: 12000, capacity: 200, load: 200 },
      ], 0.8, 0, 0, 264),
    },
  ],
  r10: [
    {
      id: 'a', name: 'Вариант А', description: '2× Газель + 1× Ларгус, +30 мин', cost: 10500, time: '+30 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 45 },
        { name: 'Ларгус', count: 1, fixedCost: 1800, capacity: 20, load: 16 },
      ], 0.8, 1.2, 30, 331),
    },
    {
      id: 'b', name: 'Вариант Б', description: '3× Газель, немедленно', cost: 13800, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 3, fixedCost: 2800, capacity: 50, load: 40 },
      ], 0.8, 0, 0, 331),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Фура, немедленно', cost: 17500, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: 331 },
      ], 0.8, 0, 0, 331),
    },
  ],
}

function defaultScenarios(forecast: number): CostScenario[] {
  return [
    {
      id: 'a', name: 'Вариант А', description: '2× Газель + 1× Ларгус, +40 мин', cost: 11200, time: '+40 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 2, fixedCost: 2800, capacity: 50, load: 45 },
        { name: 'Ларгус', count: 1, fixedCost: 1800, capacity: 20, load: 16 },
      ], 0.8, 1.2, 40, forecast),
    },
    {
      id: 'b', name: 'Вариант Б', description: '1× Газель + 2× Ларгус, +15 мин', cost: 13800, time: '+15 мин',
      breakdown: makeBreakdown([
        { name: 'Газель', count: 1, fixedCost: 2800, capacity: 50, load: 44 },
        { name: 'Ларгус', count: 2, fixedCost: 1800, capacity: 20, load: 17 },
      ], 0.8, 1.2, 15, forecast),
    },
    {
      id: 'c', name: 'Вариант В', description: '1× Фура, немедленно', cost: 18000, time: 'Сейчас',
      breakdown: makeBreakdown([
        { name: 'Фура', count: 1, fixedCost: 8500, capacity: 500, load: forecast },
      ], 0.8, 0, 0, forecast),
    },
  ]
}

export function getCostScenarios(recommendationId: string, forecast = 300): CostScenario[] {
  const scenarios = costScenariosMap[recommendationId]
  if (!scenarios) return defaultScenarios(forecast)
  return [...scenarios].sort((a, b) => a.cost - b.cost)
}

// ─── Default risk settings ────────────────────────────────────────────────────

export const defaultRiskSettings: RiskSettings = {
  economyThreshold: 65,
  maxWaitMinutes: 55,
  idleCostPerMinute: 8,
}
