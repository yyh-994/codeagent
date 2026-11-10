"""ORM 模型 — 用户、学生成绩与个人课表。"""

import hashlib, uuid as _uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, CheckConstraint, Column, Date, DateTime, DECIMAL,
    ForeignKey, Index, Integer, String, Time, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

#创建一个基类,所有 ORM 模型类都继承它，从而自动获得"映射到数据库表"的能力。
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    uuid = Column(String(36), primary_key=True, default=lambda: str(_uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role_rel = relationship("Role", back_populates="user", uselist=False, cascade="all, delete-orphan")

    #静态方法：这个函数不依赖实例状态，它就是个归类用的纯工具函数
    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def set_password(self, password: str):
        self.password_hash = self._hash(password)

    def check_password(self, password: str) -> bool:
        return self.password_hash == self._hash(password)


class Role(Base):
    __tablename__ = "roles"
    uuid = Column(String(36), ForeignKey("users.uuid"), primary_key=True)
    role = Column(String(20), nullable=False, default="basic")
    user = relationship("User", back_populates="role_rel")


class LoginAttempt(Base):
    """登录尝试记录 — 防暴力破解"""
    __tablename__ = "login_attempts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    ip = Column(String(45))
    success = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Student(Base):
    __tablename__ = "students"

    id = Column(String(20), primary_key=True)
    name = Column(String(50), nullable=False)
    gender = Column(String(4))
    age = Column(Integer)
    major = Column(String(100))
    class_name = Column(String(50))
    grade = Column(Integer)
    gpa = Column(DECIMAL(3, 1))
    phone = Column(String(20))
    email = Column(String(100))
    dormitory = Column(String(50))
    address = Column(String(200))
    father_name = Column(String(20))
    father_phone = Column(String(20))
    mother_name = Column(String(20))
    mother_phone = Column(String(20))


class Grade(Base):
    __tablename__ = "grades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(20), ForeignKey("students.id"), nullable=False)
    course = Column(String(100), nullable=False)
    score = Column(DECIMAL(5, 1))
    semester = Column(String(20))


class AcademicTerm(Base):
    """学期校历：教学周按第一周周一连续计算。"""

    __tablename__ = "academic_terms"
    __table_args__ = (
        CheckConstraint("end_date >= week1_monday", name="ck_term_dates"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    term_id = Column(String(50), primary_key=True)
    week1_monday = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)


class ClassPeriod(Base):
    """各学期共用的节次时间。"""

    __tablename__ = "class_periods"
    __table_args__ = (
        CheckConstraint("period_no > 0", name="ck_period_positive"),
        CheckConstraint("end_time > start_time", name="ck_period_times"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    period_no = Column(Integer, primary_key=True, autoincrement=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)


class CourseSchedule(Base):
    """每行表示某教学周的一次连续上课时段。"""

    __tablename__ = "course_schedule"
    __table_args__ = (
        CheckConstraint("week_no > 0", name="ck_schedule_week"),
        CheckConstraint("weekday BETWEEN 1 AND 7", name="ck_schedule_weekday"),
        CheckConstraint("end_period >= start_period", name="ck_schedule_periods"),
        UniqueConstraint(
            "term_id", "week_no", "weekday", "start_period",
            name="uq_schedule_start",
        ),
        Index("ix_schedule_day", "term_id", "week_no", "weekday"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    term_id = Column(
        String(50), ForeignKey("academic_terms.term_id", name="fk_schedule_term"),
        nullable=False,
    )
    course_name = Column(String(100), nullable=False)
    week_no = Column(Integer, nullable=False)
    weekday = Column(Integer, nullable=False)
    start_period = Column(
        Integer, ForeignKey("class_periods.period_no", name="fk_start_period"),
        nullable=False,
    )
    end_period = Column(
        Integer, ForeignKey("class_periods.period_no", name="fk_end_period"),
        nullable=False,
    )
    classroom = Column(String(100), nullable=False)
