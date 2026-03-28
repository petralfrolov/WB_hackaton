declare module 'react-simple-maps' {
  import type { ReactNode, SVGProps, MouseEvent } from 'react'

  export interface GeographyFeature {
    id: string | number
    rsmKey: string
    properties: Record<string, unknown>
    geometry: {
      type: string
      coordinates: unknown[]
    }
  }

  export interface ComposableMapProps {
    projection?: string
    projectionConfig?: {
      scale?: number
      center?: [number, number]
      rotate?: [number, number, number]
      parallels?: [number, number]
    }
    width?: number
    height?: number
    style?: React.CSSProperties
    className?: string
    children?: ReactNode
  }

  export interface GeographiesProps {
    geography: string | object
    children: (props: { geographies: GeographyFeature[] }) => ReactNode
  }

  export interface GeographyProps extends SVGProps<SVGPathElement> {
    geography: GeographyFeature
    style?: {
      default?: React.CSSProperties
      hover?: React.CSSProperties
      pressed?: React.CSSProperties
    }
  }

  export interface MarkerProps {
    coordinates: [number, number]
    children?: ReactNode
    onClick?: () => void
    onMouseEnter?: (event: MouseEvent<SVGGElement>) => void
    onMouseLeave?: (event: MouseEvent<SVGGElement>) => void
    style?: React.CSSProperties
    className?: string
  }

  export interface ZoomableGroupProps {
    zoom?: number
    center?: [number, number]
    onMoveEnd?: (position: { coordinates: [number, number]; zoom: number }) => void
    translateExtent?: [[number, number], [number, number]]
    children?: ReactNode
  }

  export function ComposableMap(props: ComposableMapProps): JSX.Element
  export function Geographies(props: GeographiesProps): JSX.Element
  export function Geography(props: GeographyProps): JSX.Element
  export function Marker(props: MarkerProps): JSX.Element
  export function ZoomableGroup(props: ZoomableGroupProps): JSX.Element
}
