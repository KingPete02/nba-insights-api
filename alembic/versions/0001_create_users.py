from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ENUM

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum safely (won't error if it already exists)
    plan_enum = ENUM("FREE", "PRO", name="plan_enum", create_type=False)
    plan_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("plan", plan_enum, nullable=False, server_default="FREE"),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("timezone('utc', now())")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("timezone('utc', now())")),
    )


def downgrade() -> None:
    op.drop_table("users")
    plan_enum = ENUM("FREE", "PRO", name="plan_enum", create_type=False)
    plan_enum.drop(op.get_bind(), checkfirst=True)
