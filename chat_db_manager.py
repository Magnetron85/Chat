import sqlite3
import json
import os
import logging
import re
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
from fuzzywuzzy import fuzz, process  # For Levenshtein distance

class ChatDatabaseManager(QObject):
    """Manages persistent storage of chat threads and messages using SQLite"""
    
    # Signals for search results and database operations
    search_results_changed = pyqtSignal(list)
    thread_list_changed = pyqtSignal(list)
    thread_loaded = pyqtSignal(dict, list)  # Thread metadata, message list
    
    def __init__(self, db_path="chat_history.db"):
        super().__init__()
        self.db_path = db_path
        self.current_thread_id = None
        
        # Initialize database if it doesn't exist
        self._initialize_database()
        
    def _initialize_database(self):
        """Create the database and tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create threads table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                provider TEXT,
                model TEXT,
                preprompt TEXT,
                is_archived INTEGER DEFAULT 0
            )
            ''')
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                provider TEXT,
                model TEXT,
                FOREIGN KEY (thread_id) REFERENCES threads (id) ON DELETE CASCADE
            )
            ''')
            
            # Create index for faster searches
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_thread ON messages (thread_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_content ON messages (content)')
            
            # Enable foreign keys
            cursor.execute('PRAGMA foreign_keys = ON')
            
            conn.commit()
            conn.close()
            logging.debug("Database initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing database: {str(e)}")
    
    def create_thread(self, title, provider=None, model=None, preprompt=None):
        """Create a new chat thread and return its ID"""
        try:
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO threads (title, created_at, last_updated, provider, model, preprompt)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, now, now, provider, model, preprompt))
            
            thread_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Set this as the current thread
            self.current_thread_id = thread_id
            
            # Emit signal that thread list has changed
            self._emit_thread_list_changed()
            
            return thread_id
        except Exception as e:
            logging.error(f"Error creating thread: {str(e)}")
            return None
    
    def get_thread(self, thread_id):
        """Get metadata for a specific thread"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, title, created_at, last_updated, provider, model, preprompt, is_archived
            FROM threads
            WHERE id = ?
            ''', (thread_id,))
            
            thread = cursor.fetchone()
            conn.close()
            
            if thread:
                return dict(thread)
            return None
        except Exception as e:
            logging.error(f"Error getting thread: {str(e)}")
            return None
    
    def update_thread(self, thread_id, title=None, provider=None, model=None, preprompt=None, is_archived=None):
        """Update thread metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query based on provided parameters
            update_parts = []
            params = []
            
            if title is not None:
                update_parts.append("title = ?")
                params.append(title)
            
            if provider is not None:
                update_parts.append("provider = ?")
                params.append(provider)
            
            if model is not None:
                update_parts.append("model = ?")
                params.append(model)
            
            if preprompt is not None:
                update_parts.append("preprompt = ?")
                params.append(preprompt)
            
            if is_archived is not None:
                update_parts.append("is_archived = ?")
                params.append(1 if is_archived else 0)
                
            # Always update last_updated
            update_parts.append("last_updated = ?")
            params.append(datetime.now().isoformat())
            
            # Add thread_id to params
            params.append(thread_id)
            
            if update_parts:
                query = f'''
                UPDATE threads
                SET {", ".join(update_parts)}
                WHERE id = ?
                '''
                
                cursor.execute(query, params)
                conn.commit()
            
            conn.close()
            
            # Emit signal that thread list has changed
            self._emit_thread_list_changed()
            
            return True
        except Exception as e:
            logging.error(f"Error updating thread: {str(e)}")
            return False
    
    def delete_thread(self, thread_id):
        """Delete a thread and all its messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete the thread (will cascade to messages)
            cursor.execute('DELETE FROM threads WHERE id = ?', (thread_id,))
            conn.commit()
            conn.close()
            
            # If this was the current thread, reset current_thread_id
            if self.current_thread_id == thread_id:
                self.current_thread_id = None
            
            # Emit signal that thread list has changed
            self._emit_thread_list_changed()
            
            return True
        except Exception as e:
            logging.error(f"Error deleting thread: {str(e)}")
            return False
    
    def get_all_threads(self, include_archived=False):
        """Get a list of all threads, ordered by last updated"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
            SELECT id, title, created_at, last_updated, provider, model, is_archived
            FROM threads
            '''
            
            if not include_archived:
                query += " WHERE is_archived = 0"
                
            query += " ORDER BY last_updated DESC"
            
            cursor.execute(query)
            threads = [dict(row) for row in cursor.fetchall()]
            
            # Add message count to each thread
            for thread in threads:
                cursor.execute('SELECT COUNT(*) FROM messages WHERE thread_id = ?', (thread['id'],))
                thread['message_count'] = cursor.fetchone()[0]
            
            conn.close()
            return threads
        except Exception as e:
            logging.error(f"Error getting all threads: {str(e)}")
            return []
    
    def add_message(self, thread_id, role, content, provider=None, model=None):
        """Add a message to a thread with the specified role"""
        try:
            # Log what we're adding
            logging.debug(f"Adding message to thread {thread_id} with role={role}, content={content[:50] if content else 'None'}...")
            
            # Validate role
            if role not in ["user", "assistant", "system"]:
                logging.warning(f"Invalid role '{role}' - defaulting to 'user'")
                role = "user"
                
            # Make sure we have content
            if not content:
                logging.warning(f"Empty content for message in thread {thread_id}")
                content = ""
                
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert the message with explicit role
            cursor.execute('''
            INSERT INTO messages (thread_id, role, content, timestamp, provider, model)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (thread_id, role, content, now, provider, model))
            
            # Update the thread's last_updated time
            cursor.execute('''
            UPDATE threads 
            SET last_updated = ?
            WHERE id = ?
            ''', (now, thread_id))
            
            conn.commit()
            
            # Verify the message was actually saved
            cursor.execute("SELECT id FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT 1", (thread_id,))
            message_id = cursor.fetchone()
            if message_id:
                logging.debug(f"Message saved successfully with id {message_id[0]}")
            else:
                logging.error("No message ID returned after insert - possible database error")
                
            conn.close()
            
            # Emit signal that thread list has changed (to update last_updated in UI)
            self._emit_thread_list_changed()
            
            return True
        except Exception as e:
            logging.error(f"Error adding message: {str(e)}")
            return False
    
    def get_messages(self, thread_id, limit=None):
        """Get all messages for a thread, ordered by timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
            SELECT id, role, content, timestamp, provider, model
            FROM messages
            WHERE thread_id = ?
            ORDER BY timestamp ASC
            '''
            
            if limit:
                query += f" LIMIT {int(limit)}"
            
            cursor.execute(query, (thread_id,))
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return messages
        except Exception as e:
            logging.error(f"Error getting messages: {str(e)}")
            return []
    
    def search_messages(self, search_term, thread_id=None, fuzzy_threshold=75, proximity_distance=5, use_fuzzy=True):
        """
        Optimized search for messages with fuzzy matching and proximity search
        that scales efficiently with large datasets.
        """
        try:
            if not search_term:
                return []
                
            logging.debug(f"Searching for messages with term: '{search_term}', thread_id: {thread_id}, fuzzy: {use_fuzzy}")
            
            # Fall back to original search if fuzzy search is disabled
            if not use_fuzzy:
                return self._search_messages_basic(search_term, thread_id)
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # ---- OPTIMIZATION 1: Pre-filter with SQL ----
            # Use SQLite's full-text search capabilities if available, 
            # otherwise construct a smart pre-filter using OR conditions
            search_words = search_term.lower().split()
            
            # Create SQL for pre-filtering (reduces the initial result set significantly)
            like_conditions = []
            params = []
            
            # Add individual word conditions 
            # (will catch messages containing at least one search word)
            for word in search_words:
                if len(word) >= 3:  # Only use words with 3+ chars for pre-filtering
                    like_conditions.append("lower(m.content) LIKE ?")
                    params.append(f"%{word}%")
            
            # Always add the full phrase as a condition
            like_conditions.append("lower(m.content) LIKE ?")
            params.append(f"%{search_term.lower()}%")
            
            # Construct the WHERE clause
            where_clause = " OR ".join(like_conditions)
            
            # Add thread constraint if specified
            thread_condition = ""
            if thread_id:
                thread_condition = "m.thread_id = ? AND "
                params.insert(0, thread_id)
            
            # Execute the query with pre-filtering
            query = f'''
            SELECT m.id, m.thread_id, m.role, m.content, m.timestamp, 
                   t.title as thread_title
            FROM messages m
            JOIN threads t ON m.thread_id = t.id
            WHERE {thread_condition}({where_clause})
            ORDER BY m.timestamp DESC
            '''
            
            cursor.execute(query, params)
            pre_filtered_results = [dict(row) for row in cursor.fetchall()]
            logging.debug(f"Pre-filtered to {len(pre_filtered_results)} potential matches")
            
            # ---- OPTIMIZATION 2: Limit expensive fuzzy processing ----
            # Set a reasonable batch size depending on query complexity
            batch_size = min(1000, max(100, 500 // len(search_words)))
            pre_filtered_results = pre_filtered_results[:batch_size]
            
            # ---- OPTIMIZATION 3: Targeted fuzzy matching ----
            filtered_results = []
            
            for result in pre_filtered_results:
                content = result['content']
                best_ratio = 0
                is_match = False
                
                # Fast check: Direct substring match (case insensitive)
                if search_term.lower() in content.lower():
                    best_ratio = 100
                    is_match = True
                
                # Check for proximity match first (cheaper than fuzzy)
                elif len(search_words) > 1:
                    # Only do proximity check if all words are present
                    if all(word.lower() in content.lower() for word in search_words):
                        is_match = self._check_proximity_optimized(content, search_words, proximity_distance)
                        if is_match:
                            best_ratio = 85  # Arbitrary score for proximity matches
                
                # Only if we haven't found a match yet, try fuzzy matching
                if not is_match:
                    # ---- OPTIMIZATION 4: Smart chunking ----
                    # Instead of all possible chunks, extract relevant sentences/paragraphs
                    relevant_chunks = self._extract_relevant_chunks(content, search_words)
                    
                    for chunk in relevant_chunks:
                        ratio = fuzz.partial_ratio(search_term.lower(), chunk.lower())
                        best_ratio = max(best_ratio, ratio)
                        
                        if best_ratio >= fuzzy_threshold:
                            is_match = True
                            break
                
                # Add to results if it passes criteria
                if is_match or best_ratio >= fuzzy_threshold:
                    result['match_score'] = best_ratio
                    result['match_type'] = 'fuzzy' if best_ratio >= fuzzy_threshold else 'proximity'
                    filtered_results.append(result)
            
            # Sort by match score
            filtered_results.sort(key=lambda x: x['match_score'], reverse=True)
            
            conn.close()
            
            # Emit signal with search results
            self.search_results_changed.emit(filtered_results)
            
            logging.debug(f"Enhanced search returned {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logging.error(f"Error in enhanced message search: {str(e)}")
            # Fall back to basic search on error
            return self._search_messages_basic(search_term, thread_id)
    
    def _search_messages_basic(self, search_term, thread_id=None):
        """Original LIKE-based search as fallback"""
        try:
            logging.debug(f"Using basic search for messages with term: '{search_term}', thread_id: {thread_id}")
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Simplify the search to make it more robust - just use a basic LIKE pattern
            search_pattern = f"%{search_term}%"
            
            if thread_id:
                # Search within a specific thread
                query = '''
                SELECT m.id, m.thread_id, m.role, m.content, m.timestamp, 
                       t.title as thread_title
                FROM messages m
                JOIN threads t ON m.thread_id = t.id
                WHERE m.thread_id = ? AND lower(m.content) LIKE lower(?)
                ORDER BY m.timestamp DESC
                '''
                cursor.execute(query, (thread_id, search_pattern))
            else:
                # Search across all threads
                query = '''
                SELECT m.id, m.thread_id, m.role, m.content, m.timestamp, 
                       t.title as thread_title
                FROM messages m
                JOIN threads t ON m.thread_id = t.id
                WHERE lower(m.content) LIKE lower(?)
                ORDER BY m.timestamp DESC
                '''
                cursor.execute(query, (search_pattern,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            # Emit signal with search results
            self.search_results_changed.emit(results)
            
            return results
        except Exception as e:
            logging.error(f"Error in basic message search: {str(e)}")
            return []
    
    def search_threads(self, search_term, fuzzy_threshold=75, proximity_distance=5, use_fuzzy=True):
        """
        Optimized search for threads with enhanced matching that scales efficiently.
        """
        try:
            if not search_term:
                return []
                
            logging.debug(f"Searching for threads with term: '{search_term}', fuzzy: {use_fuzzy}")
            
            # Fall back to original search if fuzzy search is disabled
            if not use_fuzzy:
                return self._search_threads_basic(search_term)
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Optimized approach: use SQL to pre-filter both titles and content
            search_words = search_term.lower().split()
            
            # ---- OPTIMIZATION 1: Pre-filter with SQL ----
            # First get potential thread matches by title (typically smaller set)
            title_conditions = []
            title_params = []
            
            # Add direct title match
            title_conditions.append("lower(title) LIKE ?")
            title_params.append(f"%{search_term.lower()}%")
            
            # Add individual word conditions for title
            for word in search_words:
                if len(word) >= 3:  # Only use words with 3+ chars
                    title_conditions.append("lower(title) LIKE ?")
                    title_params.append(f"%{word}%")
                    
            title_where = " OR ".join(title_conditions)
            
            query = f'''
            SELECT id, title, created_at, last_updated, provider, model, is_archived
            FROM threads
            WHERE {title_where}
            '''
            
            cursor.execute(query, title_params)
            title_candidate_threads = [dict(row) for row in cursor.fetchall()]
            
            # ---- OPTIMIZATION 2: Fuzzy title matching ----
            thread_results = {}
            
            for thread in title_candidate_threads:
                thread_id = thread['id']
                title = thread['title']
                
                # Direct match scores highest
                if search_term.lower() in title.lower():
                    thread_results[thread_id] = {
                        **thread,
                        'found_by_title': True,
                        'title_score': 100,
                        'found_by_content': False,
                        'content_score': 0,
                        'matching_messages': []
                    }
                else:
                    # Apply fuzzy matching to title
                    title_score = fuzz.partial_ratio(search_term.lower(), title.lower())
                    
                    if title_score >= fuzzy_threshold:
                        thread_results[thread_id] = {
                            **thread,
                            'found_by_title': True,
                            'title_score': title_score,
                            'found_by_content': False,
                            'content_score': 0,
                            'matching_messages': []
                        }
            
            # ---- OPTIMIZATION 3: Targeted content searching ----
            # Construct message content search conditions
            content_conditions = []
            content_params = []
            
            # Add direct content match
            content_conditions.append("lower(m.content) LIKE ?")
            content_params.append(f"%{search_term.lower()}%")
            
            # Add individual word conditions for content
            for word in search_words:
                if len(word) >= 3:  # Only use words with 3+ chars
                    content_conditions.append("lower(m.content) LIKE ?")
                    content_params.append(f"%{word}%")
            
            content_where = " OR ".join(content_conditions)
            
            # Get threads with matching message content
            query = f'''
            SELECT DISTINCT t.id, t.title, t.created_at, t.last_updated, t.provider, 
                   t.model, t.is_archived, m.id as message_id, m.content
            FROM threads t
            JOIN messages m ON t.id = m.thread_id
            WHERE {content_where}
            LIMIT 2000  -- Reasonable cap on result size
            '''
            
            cursor.execute(query, content_params)
            content_matches = [dict(row) for row in cursor.fetchall()]
            
            # Group messages by thread for efficient processing
            thread_msgs = {}
            for match in content_matches:
                thread_id = match['id']
                if thread_id not in thread_msgs:
                    thread_msgs[thread_id] = []
                thread_msgs[thread_id].append({
                    'id': match['message_id'],
                    'content': match['content']
                })
            
            # Process each thread's messages
            for thread_id, messages in thread_msgs.items():
                # Avoid redundant work for threads already matched by title
                if thread_id in thread_results and len(messages) > 20:
                    # If we already matched by title and there are many messages,
                    # just mark it as matched by content too without detailed scoring
                    thread_results[thread_id]['found_by_content'] = True
                    thread_results[thread_id]['content_score'] = 90
                    thread_results[thread_id]['matching_messages'] = [{'id': msg['id'], 'score': 90} for msg in messages[:5]]
                    continue
                    
                # Process content matches (up to a reasonable limit)
                msgs_to_process = messages[:min(len(messages), 50)]
                matching_msgs = []
                
                for msg in msgs_to_process:
                    content = msg['content']
                    best_ratio = 0
                    
                    # Direct substring match (case insensitive)
                    if search_term.lower() in content.lower():
                        best_ratio = 100
                    # Proximity match
                    elif len(search_words) > 1 and all(word.lower() in content.lower() for word in search_words):
                        if self._check_proximity_optimized(content, search_words, proximity_distance):
                            best_ratio = 85
                    # Fuzzy match as last resort (most expensive)
                    else:
                        relevant_chunks = self._extract_relevant_chunks(content, search_words)
                        for chunk in relevant_chunks:
                            ratio = fuzz.partial_ratio(search_term.lower(), chunk.lower())
                            best_ratio = max(best_ratio, ratio)
                            if best_ratio >= fuzzy_threshold:
                                break
                    
                    # Store matching messages
                    if best_ratio >= fuzzy_threshold:
                        matching_msgs.append({
                            'id': msg['id'],
                            'score': best_ratio
                        })
                        
                # If we found matching messages, add or update thread results
                if matching_msgs:
                    best_content_score = max([m['score'] for m in matching_msgs])
                    
                    if thread_id in thread_results:
                        thread_results[thread_id]['found_by_content'] = True
                        thread_results[thread_id]['content_score'] = best_content_score
                        thread_results[thread_id]['matching_messages'] = matching_msgs
                    else:
                        # Thread wasn't found by title but has matching content
                        # Get thread data first
                        cursor.execute('''
                        SELECT id, title, created_at, last_updated, provider, model, is_archived
                        FROM threads WHERE id = ?
                        ''', (thread_id,))
                        thread_data = dict(cursor.fetchone())
                        
                        thread_results[thread_id] = {
                            **thread_data,
                            'found_by_title': False,
                            'title_score': 0,
                            'found_by_content': True,
                            'content_score': best_content_score,
                            'matching_messages': matching_msgs
                        }
            
            # Convert to list and add message counts
            results = list(thread_results.values())
            
            # Sort by best match score (either title or content)
            results.sort(key=lambda x: max(x.get('title_score', 0), x.get('content_score', 0)), reverse=True)
            
            # Add message counts
            for thread in results:
                cursor.execute('SELECT COUNT(*) FROM messages WHERE thread_id = ?', (thread['id'],))
                thread['message_count'] = cursor.fetchone()[0]
                thread['matching_message_count'] = len(thread.get('matching_messages', []))
            
            conn.close()
            return results
            
        except Exception as e:
            logging.error(f"Error in enhanced thread search: {str(e)}")
            # Fall back to basic search on error
            return self._search_threads_basic(search_term)
            
    def _extract_relevant_chunks(self, content, search_words, max_chunks=10):
        """
        Intelligently extract the most relevant text chunks based on search words.
        Much more efficient than processing all possible chunks.
        """
        # If content is too short, just return it
        if len(content) < 200:
            return [content]
        
        # Find sentences or paragraphs containing search words
        chunks = []
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Find sentences containing any search word
        relevant_sentences = []
        for sentence in sentences:
            if any(word.lower() in sentence.lower() for word in search_words):
                relevant_sentences.append(sentence)
        
        # If we found relevant sentences, use them
        if relevant_sentences:
            chunks.extend(relevant_sentences[:max_chunks])
        
        # If we don't have enough chunks yet, also check paragraphs
        if len(chunks) < max_chunks // 2:
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if any(word.lower() in para.lower() for word in search_words):
                    # Don't add if it's too similar to what we already have
                    if not any(fuzz.ratio(para, chunk) > 70 for chunk in chunks):
                        chunks.append(para)
                        if len(chunks) >= max_chunks:
                            break
        
        # If we still don't have enough, add some context from nearby sentences
        if len(relevant_sentences) > 0 and len(chunks) < max_chunks:
            for i, sentence in enumerate(sentences):
                if sentence in relevant_sentences:
                    # Add previous sentence as context if available
                    if i > 0 and sentences[i-1] not in chunks:
                        chunks.append(sentences[i-1] + " " + sentence)
                    # Add next sentence as context if available
                    if i < len(sentences) - 1 and sentences[i+1] not in chunks:
                        chunks.append(sentence + " " + sentences[i+1])
                    
                    if len(chunks) >= max_chunks:
                        break
        
        # If we somehow ended up with no chunks, fall back to the most likely
        # parts of the content based on word windows
        if not chunks:
            words = content.split()
            # Find regions with search words
            for i, word in enumerate(words):
                if any(search_word.lower() in word.lower() for search_word in search_words):
                    # Extract a window around this word
                    start = max(0, i - 10)
                    end = min(len(words), i + 11)
                    window = ' '.join(words[start:end])
                    chunks.append(window)
                    if len(chunks) >= max_chunks:
                        break
        
        return chunks

    def _check_proximity_optimized(self, content, search_words, max_distance):
        """
        Optimized version of proximity checking that stops as soon as
        it finds a match, avoiding exponential combinations.
        """
        # Quick validation: make sure all words are present
        content_lower = content.lower()
        if not all(word.lower() in content_lower for word in search_words):
            return False
            
        # Break into words
        words = re.findall(r'\b\w+\b', content_lower)
        
        # For each word position, check if others are within range
        for i, word in enumerate(words):
            # Skip words that don't match any search word
            if not any(search_word == word for search_word in search_words):
                continue
                
            # Initialize tracking of found words
            found_words = {search_word: False for search_word in search_words}
            found_words[[sw for sw in search_words if sw == word][0]] = True
            
            # Check window around this word
            start = max(0, i - max_distance)
            end = min(len(words), i + max_distance + 1)
            
            for j in range(start, end):
                if i == j:
                    continue  # Skip the current word
                    
                # Mark any search words found in this window
                for search_word in search_words:
                    if not found_words[search_word] and search_word == words[j]:
                        found_words[search_word] = True
            
            # If all search words were found within the window, we have a match
            if all(found_words.values()):
                return True
                
        return False
    
    def _search_threads_basic(self, search_term):
        """Original LIKE-based search as fallback"""
        try:
            logging.debug(f"Using basic search for threads with term: '{search_term}'")
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Create simplified search pattern
            search_pattern = f"%{search_term}%"
            
            # First search in thread titles
            query = '''
            SELECT id, title, created_at, last_updated, provider, model, is_archived
            FROM threads
            WHERE lower(title) LIKE lower(?)
            '''
            
            cursor.execute(query, (search_pattern,))
            threads_by_title = [dict(row) for row in cursor.fetchall()]
            
            # Keep track of thread IDs found by title
            threads_by_title_ids = set(thread['id'] for thread in threads_by_title)
            
            # Then search in message content
            query = '''
            SELECT DISTINCT t.id, t.title, t.created_at, t.last_updated, t.provider, 
                   t.model, t.is_archived
            FROM threads t
            JOIN messages m ON t.id = m.thread_id
            WHERE lower(m.content) LIKE lower(?)
            '''
            
            cursor.execute(query, (search_pattern,))
            threads_by_content = [dict(row) for row in cursor.fetchall()]
            
            # Combine results, avoiding duplicates
            threads = []
            for thread in threads_by_title:
                thread['found_by_title'] = True
                thread['found_by_content'] = False
                threads.append(thread)
            
            for thread in threads_by_content:
                if thread['id'] not in threads_by_title_ids:
                    thread['found_by_title'] = False
                    thread['found_by_content'] = True
                    threads.append(thread)
                else:
                    # Mark threads found by both title and content
                    for t in threads:
                        if t['id'] == thread['id']:
                            t['found_by_content'] = True
                            break
            
            # Sort by last updated
            threads.sort(key=lambda x: x['last_updated'], reverse=True)
            
            # Add message count and matching message count
            for thread in threads:
                # Get total message count
                cursor.execute('SELECT COUNT(*) FROM messages WHERE thread_id = ?', (thread['id'],))
                thread['message_count'] = cursor.fetchone()[0]
                
                # Get matching message count
                if thread['found_by_content']:
                    cursor.execute('''
                    SELECT COUNT(*) 
                    FROM messages 
                    WHERE thread_id = ? AND lower(content) LIKE lower(?)
                    ''', (thread['id'], search_pattern))
                    thread['matching_messages'] = cursor.fetchone()[0]
            
            conn.close()
            return threads
        except Exception as e:
            logging.error(f"Error in basic thread search: {str(e)}")
            return []
    
    def _get_content_chunks(self, content, chunk_size):
        """Break content into overlapping chunks for better fuzzy matching"""
        words = content.split()
        if not words:
            return []
            
        chunks = []
        # For very short content, just return the whole thing
        if len(words) <= chunk_size + 2:
            return [content]
            
        # Create sliding window of words
        for i in range(len(words) - chunk_size + 1):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
            
        # Also add sentences as chunks since they're natural units
        sentences = re.split(r'[.!?]\s+', content)
        chunks.extend([s.strip() for s in sentences if s.strip()])
        
        return chunks
    
    def _check_proximity(self, content, search_words, max_distance):
        """
        Check if all search words appear in the content within the specified
        word distance of each other, even if not adjacent
        """
        content_lower = content.lower()
        words = re.findall(r'\b\w+\b', content_lower)
        
        # Find positions of each search word
        word_positions = {}
        for word in search_words:
            positions = []
            for i, w in enumerate(words):
                if word == w:
                    positions.append(i)
            if not positions:  # If any word is missing, no match
                return False
            word_positions[word] = positions
        
        # Check if any combination of positions satisfies the proximity requirement
        position_combinations = self._get_position_combinations(word_positions)
        
        for positions in position_combinations:
            positions.sort()
            # Check if distance between first and last position is within limit
            if positions[-1] - positions[0] <= max_distance:
                return True
                
        return False
    
    def _get_position_combinations(self, word_positions):
        """Get all combinations of positions for search words"""
        words = list(word_positions.keys())
        
        def backtrack(index, current_combination):
            if index == len(words):
                return [current_combination]
            
            result = []
            word = words[index]
            for pos in word_positions[word]:
                new_combination = current_combination + [pos]
                result.extend(backtrack(index + 1, new_combination))
            
            return result
        
        return backtrack(0, [])
    
    def export_thread(self, thread_id, file_path):
        """Export a thread to a JSON file"""
        try:
            thread = self.get_thread(thread_id)
            if not thread:
                return False
                
            messages = self.get_messages(thread_id)
            
            export_data = {
                "thread": thread,
                "messages": messages
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
                
            return True
        except Exception as e:
            logging.error(f"Error exporting thread: {str(e)}")
            return False
    
    def import_thread(self, file_path):
        """Import a thread from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
                
            if not isinstance(import_data, dict) or "thread" not in import_data or "messages" not in import_data:
                logging.error("Invalid import file format")
                return False
                
            # Create the thread
            thread_data = import_data["thread"]
            now = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO threads (title, created_at, last_updated, provider, model, preprompt, is_archived)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                thread_data.get("title", "Imported Thread"),
                thread_data.get("created_at", now),
                now,  # Always use current time for last_updated
                thread_data.get("provider"),
                thread_data.get("model"),
                thread_data.get("preprompt"),
                thread_data.get("is_archived", 0)
            ))
            
            new_thread_id = cursor.lastrowid
            
            # Import messages
            for msg in import_data["messages"]:
                cursor.execute('''
                INSERT INTO messages (thread_id, role, content, timestamp, provider, model)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    new_thread_id,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    msg.get("timestamp", now),
                    msg.get("provider"),
                    msg.get("model")
                ))
            
            conn.commit()
            conn.close()
            
            # Set as current thread
            self.current_thread_id = new_thread_id
            
            # Emit signals
            self._emit_thread_list_changed()
            
            return new_thread_id
        except Exception as e:
            logging.error(f"Error importing thread: {str(e)}")
            return False
    
    def _emit_thread_list_changed(self):
        """Helper to emit the thread_list_changed signal with current threads"""
        threads = self.get_all_threads()
        self.thread_list_changed.emit(threads)
    
    def get_conversation_history(self, thread_id):
        """Get conversation history in the format expected by LLM APIs"""
        messages = self.get_messages(thread_id)
        
        # Convert to the format used by the LLM APIs (role/content pairs)
        conversation = []
        for msg in messages:
            if msg["role"] in ["user", "assistant", "system"]:
                conversation.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Debug logging to verify history is being loaded correctly
        logging.debug(f"Loaded conversation history for thread {thread_id}: {len(conversation)} messages")
        
        return conversation