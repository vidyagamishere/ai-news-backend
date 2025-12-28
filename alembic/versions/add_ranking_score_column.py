"""add ranking score column

Revision ID: add_ranking_score_001
Revises: <previous_revision_id>
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_ranking_score_001'
down_revision = '<previous_revision_id>'  # Replace with your last migration ID
branch_labels = None
depends_on = None


def upgrade():
    # Store publisher + significance component (static, updated periodically)
    op.add_column('articles', sa.Column('static_score_component', sa.Float(), nullable=True, server_default='0.375'))
    
    # Store final ranking score (updated dynamically in query)
    op.add_column('articles', sa.Column('ranking_score', sa.Float(), nullable=True, server_default='0.5'))
    
    # Create index for performance (ranking_score will be updated in queries)
    op.create_index('idx_articles_ranking_score', 'articles', ['ranking_score', 'published_date'], postgresql_using='btree')
    
    # Add last_score_update timestamp
    op.add_column('articles', sa.Column('last_score_update', sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_index('idx_articles_ranking_score', table_name='articles')
    op.drop_column('articles', 'last_score_update')
    op.drop_column('articles', 'ranking_score')
    op.drop_column('articles', 'static_score_component')
