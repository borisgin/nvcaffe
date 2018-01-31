#ifdef USE_LMDB
#include "caffe/util/db_lmdb.hpp"

#include <sys/stat.h>

#include <string>

namespace caffe { namespace db {

void LMDB::Open(const string& source, Mode mode) {
  MDB_CHECK(mdb_env_create(&mdb_env_));
  if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
  }
  int flags = 0;
  if (mode == READ) {
    flags = MDB_RDONLY | MDB_NOTLS | MDB_NOMEMINIT | MDB_NOLOCK;
  }
#ifdef __ARM_ARCH
  // Tegra
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, 1073741824UL));
//#else
//  size_t map_size = 1024UL;// * 1024UL;
//  for (int i = 0; i < 32; ++i) {
//    MDB_CHECK(mdb_env_set_mapsize(mdb_env_, map_size));
//    if (mdb_env_open(mdb_env_, source.c_str(), flags, 0664) == MDB_SUCCESS) {
//      break;
//    }
//    map_size *= 2UL;
//    std::cout << i << " " << map_size << std::endl;
//  }
#endif
  MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
//  MDB_envinfo stat;
//  MDB_CHECK(mdb_env_info(mdb_env_, &stat));
  LOG(INFO) << "Opened lmdb " << source;
}

LMDBCursor* LMDB::NewCursor() {
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction() {
  return new LMDBTransaction(mdb_env_);
}

void LMDBTransaction::Put(const string& key, const string& value) {
  keys.push_back(key);
  values.push_back(value);
}

void LMDBTransaction::Commit() {
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Initialize MDB variables
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

  bool out_of_memory = false;
  for (int i = 0; i < keys.size(); i++) {
    mdb_key.mv_size = keys[i].size();
    mdb_key.mv_data = const_cast<char*>(keys[i].data());
    mdb_data.mv_size = values[i].size();
    mdb_data.mv_data = const_cast<char*>(values[i].data());

    int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
    if (put_rc == MDB_MAP_FULL) {
      out_of_memory = true;
      break;
    } else {
      // Failed for some other reason
      MDB_CHECK(put_rc);
    }
  }

  if (!out_of_memory) {
    // Commit the transaction
    MDB_CHECK(mdb_txn_commit(mdb_txn));
    mdb_dbi_close(mdb_env_, mdb_dbi);
    keys.clear();
    values.clear();
  } else {
    // Double the map size and retry
    mdb_txn_abort(mdb_txn);
    mdb_dbi_close(mdb_env_, mdb_dbi);
    DoubleMapSize();
    Commit();
  }
}

void LMDBTransaction::DoubleMapSize() {
  struct MDB_envinfo current_info;
  MDB_CHECK(mdb_env_info(mdb_env_, &current_info));
  size_t new_size = current_info.me_mapsize * 2;
  DLOG(INFO) << "Doubling LMDB map size to " << (new_size>>20) << "MB ...";
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, new_size));
}

}  // namespace db
}  // namespace caffe
#endif  // USE_LMDB
