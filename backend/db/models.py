"""SQLAlchemy ORM models for the transport optimizer database."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class Warehouse(Base):
    __tablename__ = "warehouses"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    office_from_id = Column(String, nullable=True)
    is_mock = Column(Boolean, default=False, nullable=False)

    routes_from = relationship("Route", foreign_keys="Route.from_warehouse_id", back_populates="from_warehouse")
    routes_to = relationship("Route", foreign_keys="Route.to_warehouse_id", back_populates="to_warehouse")
    vehicles = relationship("WarehouseVehicle", back_populates="warehouse", cascade="all, delete-orphan")
    incoming_vehicles = relationship("IncomingVehicle", back_populates="warehouse", cascade="all, delete-orphan")
    dispatch_results = relationship("DispatchResult", back_populates="warehouse", cascade="all, delete-orphan")


class Route(Base):
    __tablename__ = "routes"

    id = Column(String, primary_key=True)
    from_warehouse_id = Column(String, ForeignKey("warehouses.id"), nullable=False)
    to_warehouse_id = Column(String, ForeignKey("warehouses.id"), nullable=False)
    distance_km = Column(Float, nullable=False)
    ready_to_ship = Column(Integer, default=0)

    from_warehouse = relationship("Warehouse", foreign_keys=[from_warehouse_id], back_populates="routes_from")
    to_warehouse = relationship("Warehouse", foreign_keys=[to_warehouse_id], back_populates="routes_to")


class VehicleType(Base):
    __tablename__ = "vehicle_types"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_type = Column(String, unique=True, nullable=False)
    capacity_units = Column(Float, nullable=False)
    cost_per_km = Column(Float, nullable=False)
    underload_penalty = Column(Float, default=0.0)
    fixed_dispatch_cost = Column(Float, default=0.0)

    warehouse_vehicles = relationship("WarehouseVehicle", back_populates="vehicle_type_rel", cascade="all, delete-orphan")
    incoming_records = relationship("IncomingVehicle", back_populates="vehicle_type_rel", cascade="all, delete-orphan")


class WarehouseVehicle(Base):
    __tablename__ = "warehouse_vehicles"
    __table_args__ = (
        UniqueConstraint("warehouse_id", "vehicle_type_id", name="uq_wh_vtype"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    warehouse_id = Column(String, ForeignKey("warehouses.id"), nullable=False)
    vehicle_type_id = Column(Integer, ForeignKey("vehicle_types.id"), nullable=False)
    available = Column(Integer, nullable=False, default=0)

    warehouse = relationship("Warehouse", back_populates="vehicles")
    vehicle_type_rel = relationship("VehicleType", back_populates="warehouse_vehicles")


class IncomingVehicle(Base):
    __tablename__ = "incoming_vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    warehouse_id = Column(String, ForeignKey("warehouses.id"), nullable=False)
    horizon_idx = Column(Integer, nullable=False)
    vehicle_type_id = Column(Integer, ForeignKey("vehicle_types.id"), nullable=False)
    count = Column(Integer, nullable=False)

    warehouse = relationship("Warehouse", back_populates="incoming_vehicles")
    vehicle_type_rel = relationship("VehicleType", back_populates="incoming_records")


class DispatchResult(Base):
    __tablename__ = "dispatch_results"
    __table_args__ = (
        UniqueConstraint("warehouse_id", "timestamp", name="uq_dispatch_wh_ts"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    warehouse_id = Column(String, ForeignKey("warehouses.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    granularity_05 = Column(Text, nullable=True)  # JSON string for granularity 0.5
    granularity_1 = Column(Text, nullable=True)   # JSON string for granularity 1.0
    granularity_2 = Column(Text, nullable=True)   # JSON string for granularity 2.0
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    warehouse = relationship("Warehouse", back_populates="dispatch_results")


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)  # JSON-encoded value
