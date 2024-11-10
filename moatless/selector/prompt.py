def format_examples(examples):
    formatted = []
    for i, example in enumerate(examples, 1):
        header = f"Example {i}:\n\n"
        node_id = f"<node_id>: {i}\n"
        feedback = f"<feedback>: {example}"
        formatted.append(header + node_id + feedback)
    return "\n\n".join(formatted)

SYSTEM_PROMPT = """You are an AI tasked with analyzing a Monte Carlo Tree Search (MCTS) tree and selecting the most promising node for expansion.

When analyzing nodes:
- Consider reward, visits, and action potential.
- Prioritize nodes with higher rewards.
- Provide specific code context for the software developer agent.
- Describe nodes in relation to others (siblings, parents, etc.), not by numbers.
- Focus on context and actions, not rewards or visit counts.
- Aim for diverse 'finished' nodes through depth-wise expansion.
- Avoid loops with repetitive actions.
- Try new approaches by expanding earlier nodes if current paths aren't working, by for example going to earlier nodes in the tree and suggesting a completely new approach.

The goal is to help the software engineer efficiently find diverse, high-quality solutions through effective tree exploration."""

EXAMPLES = """The trajectory has used FindFunction 4 times in a row examining Django's query compiler:
- django/db/models/sql/compiler.py
- django/db/models/sql/query.py
- django/db/models/query.py

While we've found several query compilation methods, continuing with function searches is becoming redundant.

Recommended Next Step:
Use SemanticSearch with:
- Query: "django subquery optimization compiler"
- Category: "implementation"
- File pattern: "django/db/models/sql/*.py"

This will help identify optimization opportunities in the query compilation process rather than finding more compiler functions.
"""

EXAMPLES_1 = """
The trajectory you are on has thus far worked on the following files:
- db/models/query.py
- db/models/lookups.py
- db/models/expressions.py

and particularly focused on optimizing Django's query execution:
```python
def get_compiler(self, using=None, connection=None):
    if connection is None:
        connection = connections[using or DEFAULT_DB_ALIAS]
    return connection.ops.compiler(self.compiler)(self, connection, using)
```

Sibling nodes explored these approaches:
- Node 2: Added basic query caching:
```python
def get_compiler(self, using=None, connection=None):
    cache_key = self._generate_cache_key()
    if cache_key in self._compiler_cache:
        return self._compiler_cache[cache_key]
    connection = connection or connections[using or DEFAULT_DB_ALIAS]
    return connection.ops.compiler(self.compiler)(self, connection, using)
```
- Node 3: Implemented compiler pooling:
```python
def get_compiler(self, using=None, connection=None):
    if connection is None:
        connection = connections[using or DEFAULT_DB_ALIAS]
    compiler = self._compiler_pool.get(connection.alias)
    if compiler is None:
        compiler = connection.ops.compiler(self.compiler)(self, connection, using)
    return compiler
```

The caching approach showed a 25% speedup for repeated queries but needs refinement
for cache invalidation and memory management.

Git diff summary:
```diff
+ cache_key = self._generate_cache_key()
+ if cache_key in self._compiler_cache:
+     return self._compiler_cache[cache_key]
```

Recommended Next Step:
1. Build upon the current caching approach by:
   - Adding time-based cache invalidation
   - Implementing LRU cache with size limits
   - Adding cache key versioning for schema changes
"""

EXAMPLES_2 = """
The trajectory shows 3 consecutive RequestMoreContext actions examining scikit-learn's gradient boosting implementation:
- sklearn/ensemble/_gb.py
- sklearn/ensemble/gradient_boosting.py
- sklearn/tree/_tree.py

We now have extensive context about the gradient boosting internals, but more context requests aren't yielding new insights.

Recommended Next Step:
Use FindClass with:
- Class name: "BaseGradientBoosting"
- File pattern: "sklearn/ensemble/*.py"
To focus on the base implementation rather than gathering more peripheral context.
"""

EXAMPLES_3 = """
The trajectory has completed two different implementation attempts:

Finished Path 1 - Redis Cache Implementation:
```diff
# django/core/cache/backends/redis.py
- def get_redis_connection(self):
-     return self._client.get_connection()
+ def get_redis_connection(self):
+     if self._connection_pool is None:
+         self._connection_pool = redis.ConnectionPool(
+             host=self._options.get('HOST', 'localhost'),
+             port=self._options.get('PORT', 6379),
+             db=self._options.get('DB', 0),
+             max_connections=self._options.get('MAX_CONNECTIONS', 100)
+         )
+     return redis.Redis(connection_pool=self._connection_pool)
```

Finished Path 2 - Cache Key Optimization:
```diff
# django/core/cache/backends/base.py
- def make_key(self, key, version=None):
-     return f"{self.key_prefix}:{version}:{key}"
+ def make_key(self, key, version=None):
+     if callable(key):
+         key = key()
+     return hashlib.md5(
+         f"{self.key_prefix}:{version}:{key}".encode()
+     ).hexdigest()
```

Both approaches showed limitations:
1. Connection pooling improved performance but didn't solve memory issues
2. Key optimization reduced storage but increased CPU usage

Recommended Next Step:
Use SemanticSearch with:
- Query: "django memcached sharding implementation"
- Category: "implementation"
- File pattern: "django/core/cache/backends/memcached/*.py"

This represents a complete strategy shift to explore distributed caching with memcached sharding, rather than continuing to optimize Redis or key generation approaches.
"""

EXAMPLES_4 = """
The trajectory shows several similar cross-validation implementations in scikit-learn:

Recent Path 1 (Reward: 65):
```diff
# sklearn/model_selection/_split.py
+ def split(self, X, y=None, groups=None):
+     indices = np.arange(len(X))
+     for fold_idx in range(self.n_splits):
+         yield self._iter_test_indices(indices, fold_idx)
```

Recent Path 2 (Failing tests):
```diff
# sklearn/model_selection/_split.py
+ def split(self, X, y=None, groups=None):
+     indices = np.arange(len(X))
+     shuffled = self.random_state.permutation(indices)
+     chunk_size = len(X) // self.n_splits
+     for fold_idx in range(self.n_splits):
+         start = fold_idx * chunk_size
+         # Test failure: IndexError when n_splits doesn't divide len(X) evenly
+         yield shuffled[start:start + chunk_size], shuffled[start + chunk_size:start + 2*chunk_size]
```

Recent Path 3 (Reward: 75):
```diff
# sklearn/model_selection/_split.py
+ def split(self, X, y=None, groups=None):
+     indices = np.arange(len(X))
+     shuffled = self.random_state.permutation(indices)
+     for fold_idx in range(self.n_splits):
+         yield self._iter_test_indices(shuffled, fold_idx)
```

Notice the pattern: We're making minor variations to the same fold-based splitting approach. 
Path 2 failed due to edge cases, while the others work but with mediocre rewards (60-70 range). 
The working solutions are too similar to expect significant improvements.

Recommended Next Step:
Use SemanticSearch with:
- Query: "bootstrap sampling cross validation implementation"
- Category: "implementation"
- File pattern: "sklearn/model_selection/*.py"

Rationale: Rather than continuing with slight variations of fold-based splitting 
(which either fail on edge cases or yield mediocre rewards), we should explore 
bootstrap sampling - a fundamentally different approach that might yield higher 
rewards through better variance reduction and sampling efficiency.
"""

EXAMPLES_5 = """
The trajectory has used RequestMoreContext and FindFunction actions repeatedly to understand Django's URL resolver:
- django/urls/resolvers.py (via RequestMoreContext)
- django/urls/conf.py (via RequestMoreContext)
- django/urls/base.py (via FindFunction)

Key signs we're stuck in a search loop:
1. We've gathered context about URLResolver, URLPattern, and RegexPattern classes
2. We understand the resolve() method implementation
3. Additional context requests are returning overlapping information
4. We have a clear picture of where changes need to be made

This pattern of repeatedly gathering context without moving to implementation is a common trap.

Recommended Next Step:
Use RequestCodeChange with:
- File: "django/urls/resolvers.py"
- Change: "Add support for custom path converters in URLResolver.resolve() method:
```python
def resolve(self, path):
    # Add converter handling
    if hasattr(path, 'converter'):
        path = path.converter.to_url(path)
    return super().resolve(path)
```"

Remember: When you have sufficient understanding of the codebase through searching and context gathering, the next step should be to request concrete code changes rather than continuing to search.
"""

EXAMPLES_6 = """
The trajectory has successfully implemented Django's async database query optimization:

Finished Path - Async Query Execution:
```diff
# django/db/models/query.py
- def get_compiler(self, using=None, connection=None):
-     if connection is None:
-         connection = connections[using or DEFAULT_DB_ALIAS]
-     return connection.ops.compiler(self.compiler)(self, connection, using)

+ async def get_compiler(self, using=None, connection=None):
+     if connection is None:
+         connection = await connections.aio_get_connection(using or DEFAULT_DB_ALIAS)
+     
+     compiler = connection.ops.compiler(self.compiler)
+     compiler_instance = compiler(self, connection, using)
+     compiler_instance.async_mode = True
+     return compiler_instance

# django/db/backends/postgresql/operations.py
+ async def execute_wrapper(self, cursor, sql, params=None):
+     if self.async_mode:
+         return await cursor.execute(sql, params)
+     return cursor.execute(sql, params)
```

Implementation shows excellent results:
1. Performance tests show 3x throughput improvement
2. Memory usage reduced by 40%
3. All test cases pass, including edge cases
4. Backward compatible with sync code
5. No significant tradeoffs identified

Recommended Next Step:
Use Finish with:
- Finish reason: "Async query optimization implementation is complete with demonstrable improvements:
  - Major performance gains verified
  - Memory efficiency achieved
  - Full test coverage
  - Maintains compatibility
  - Clean implementation without significant downsides
  Further optimization would likely yield diminishing returns and risk destabilizing a well-functioning solution."
"""

EXAMPLES_4_1 = """
The trajectory shows attempts to optimize Django's query ordering logic:

Recent Path 1 (Reward: 72):
```diff
# django/db/models/sql/compiler.py
- elif not self.query.default_ordering:
-     ordering = self.query.order_by
  elif self.query.order_by:
      ordering = self.query.order_by
+ elif not self.query.default_ordering:
+     ordering = self.query.order_by
```

Recent Path 2 (Reward: 68):
```diff
# django/db/models/sql/compiler.py
  def get_ordering(self):
+     if self.query.order_by and not self.query.default_ordering:
+         return self.query.order_by
      if self.query.extra_order_by:
          return self.query.extra_order_by
```

Current approaches are stuck in a pattern: rearranging conditionals in the main ordering logic.
While these changes work, they're achieving mediocre rewards (65-75 range) and seem to just 
shuffle the same logic around.

Recommended Next Step:
Use FindFunction with:
- Function name: "find_ordering_name"
- File pattern: "django/db/models/sql/compiler.py"

Rationale: Instead of continuing to rearrange high-level ordering conditions, we should 
explore how field names are handled in relation ordering. This could reveal opportunities 
for improvement in how we process field attributes and piece together ordering clauses, 
particularly for related models.
"""

# Store all examples in a list
EXAMPLE_LIST = [
    # EXAMPLES_1,
    EXAMPLES_2,
    EXAMPLES_3,
    EXAMPLES_4_1,
    EXAMPLES_5,
    EXAMPLES_6,
]

# Generate the combined examples
ALL_EXAMPLES = format_examples(EXAMPLE_LIST)