import aiosqlite
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "chat_history.db"
MAX_ROWS = 10 # This will now store 5 user queries and 5 assistant answers


CREATE_QUESTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    query    TEXT,
    role     TEXT,
    created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# The users table remains the same
CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT,
    google_id TEXT UNIQUE,
    is_active BOOLEAN DEFAULT 1
);
"""


async def _get_conn():
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute("PRAGMA foreign_keys = ON;")
    await conn.execute(CREATE_USERS_TABLE_SQL)
    await conn.execute(CREATE_QUESTIONS_TABLE_SQL)
    await conn.commit()
    return conn


# --- CHANGE 2: Updated save_query to handle the role ---
async def save_query(user_id: str, query: str, role: str = "user"):
    conn = await _get_conn()
    await conn.execute(
        "INSERT INTO questions (user_id, query, role) VALUES (?, ?, ?)",
        (user_id, query, role),
    )
    await conn.execute(
        """
        DELETE FROM questions
        WHERE id NOT IN (
            SELECT id FROM questions
            WHERE user_id = ?
            ORDER BY created DESC
            LIMIT ?
        )
        AND user_id = ?;
        """,
        (user_id, MAX_ROWS, user_id),
    )
    await conn.commit()
    await conn.close()


# --- CHANGE 3: Updated get_recent to return the role ---
async def get_recent(user_id: str, limit: int = MAX_ROWS) -> List[Dict[str, str]]:
    conn = await _get_conn()
    cur = await conn.execute(
        """
        SELECT role, query
        FROM questions
        WHERE user_id = ?
        ORDER BY created ASC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = await cur.fetchall()
    await conn.close()
    # This now returns the dictionary format that agent.py expects, fixing the error
    return [{"role": r[0] if r[0] else "user", "query": r[1]} for r in rows]



async def get_user_by_email(email: str) -> Optional[Dict]:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = await cursor.fetchone()
    await conn.close()
    return dict(user) if user else None

async def create_user(email: str, hashed_password: str) -> Dict:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(
        "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
        (email, hashed_password)
    )
    await conn.commit()
    user_id = cursor.lastrowid


    cursor = await conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    new_user = await cursor.fetchone()
    await conn.close()
    return dict(new_user)