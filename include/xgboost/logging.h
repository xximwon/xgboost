/**
 * Copyright (c) 2015-2023 by Contributors
 * \file logging.h
 *
 * \brief defines console logging options for xgboost.  Use to enforce unified print
 *  behavior.
 */
#ifndef XGBOOST_LOGGING_H_
#define XGBOOST_LOGGING_H_

#include <xgboost/base.h>

#include <functional>  // for less
#include <memory>      // for unique_ptr
#include <mutex>
#include <sstream>
#include <string>  // for string
#include <utility>

namespace xgboost {
/**
 * @brief Base class for the global logger.
 */
class BaseLogger {
 public:
  BaseLogger();
  [[nodiscard]] std::ostream& stream() { return log_stream_; }  // NOLINT

 protected:
  std::ostringstream log_stream_;
};

class ConsoleLogger : public BaseLogger {
 public:
  enum class LogVerbosity {
    kSilent = 0,
    kWarning = 1,
    kInfo = 2,   // information may interests users.
    kDebug = 3,  // information only interesting to developers.
    kIgnore = 4  // ignore global setting
  };
  using LV = LogVerbosity;

 private:
  LogVerbosity cur_verbosity_;

 public:
  static void Configure(Args const& args);

  static LogVerbosity GlobalVerbosity();
  static LogVerbosity DefaultVerbosity();
  static bool ShouldLog(LogVerbosity verbosity);

  ConsoleLogger() = delete;
  explicit ConsoleLogger(LogVerbosity cur_verb);
  ConsoleLogger(const std::string& file, int line, LogVerbosity cur_verb);
  ~ConsoleLogger();
};

class TrackerLogger : public BaseLogger {
 public:
  ~TrackerLogger();
};

// custom logging callback; disabled for R wrapper
#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
class LogCallbackRegistry {
 public:
  using Callback = void (*)(const char*);

  LogCallbackRegistry();
  LogCallbackRegistry(LogCallbackRegistry const& that) = delete;
  LogCallbackRegistry(LogCallbackRegistry&& that) = delete;
  LogCallbackRegistry& operator=(LogCallbackRegistry const& that) = delete;
  LogCallbackRegistry& operator=(LogCallbackRegistry&& that) = delete;

  void Register(Callback log_callback) {
    std::scoped_lock lock{mu_};
    this->log_callback_ = log_callback;
  }

  [[nodiscard]] static LogCallbackRegistry& Singleton() {
    static LogCallbackRegistry callback;
    return callback;
  }

  void operator()(char const* msg) const {
    auto callback = this->Get();
    callback(msg);
  }

 private:
  [[nodiscard]] Callback Get() const {
    std::scoped_lock lock{mu_};
    // The lock is released right after returning, therefore multiple threads can run the
    // callback concurrently. The lock only protects threads from setting and getting the
    // callback at the same time.
    return log_callback_;
  }

  mutable std::mutex mu_;
  Callback log_callback_;
};
#else
class LogCallbackRegistry {
 public:
  using Callback = void (*)(const char*);
  LogCallbackRegistry() {}
  inline void Register(Callback log_callback) {}
  inline Callback Get() const {
    return nullptr;
  }

  static LogCallbackRegistry& Singleleton() {
    static LogCallbackRegistry callback;
    return callback;
  }
};
#endif  // !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0

// Redefines LOG_WARNING for controling verbosity
#if defined(LOG_WARNING)
#undef  LOG_WARNING
#endif  // defined(LOG_WARNING)

#define XGB_LOG_WARNING                                                            \
  if (::xgboost::ConsoleLogger::ShouldLog(::xgboost::ConsoleLogger::LV::kWarning)) \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__, ::xgboost::ConsoleLogger::LogVerbosity::kWarning)

// Redefines LOG_INFO for controling verbosity
#if defined(LOG_INFO)
#undef  LOG_INFO
#endif  // defined(LOG_INFO)

#define XGB_LOG_INFO                                                               \
  if (::xgboost::ConsoleLogger::ShouldLog(                                     \
          ::xgboost::ConsoleLogger::LV::kInfo))                                \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__,                                 \
                           ::xgboost::ConsoleLogger::LogVerbosity::kInfo)

#if defined(LOG_DEBUG)
#undef LOG_DEBUG
#endif  // defined(LOG_DEBUG)

#define XGB_LOG_DEBUG                                                            \
  if (::xgboost::ConsoleLogger::ShouldLog(::xgboost::ConsoleLogger::LV::kDebug)) \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__, ::xgboost::ConsoleLogger::LogVerbosity::kDebug)

// Enable LOG(CONSOLE) for print messages to console.
#define XGB_LOG_CONSOLE ::xgboost::ConsoleLogger(::xgboost::ConsoleLogger::LogVerbosity::kIgnore)

// Enable LOG(TRACKER) for print messages to tracker
#define XGB_LOG_TRACKER ::xgboost::TrackerLogger()

class LogMessageFatal {
 public:
  LogMessageFatal(char const* file, std::int32_t line);
  [[nodiscard]] std::ostream& stream();  // NOLINT
};

#if defined(CHECK)
#undef CHECK
#endif  // defined(CHECK)

#define CHECK(cond)                   \
  if (XGBOOST_EXPECT(!(cond), false)) \
  ::xgboost::LogMessageFatal(__FILE__, __LINE__).stream() << "Check failed: " #cond << ": "

#if defined(LOG_FATAL)
#undef LOG_FATAL
#endif  // defined(LOG_FATAL)

#define XGB_LOG_FATAL ::xgboost::LogMessageFatal(__FILE__, __LINE__)

template <typename X, typename Y>
std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise.
   * NOLINT(*) */
  os << " (" << x << " vs. " << y << ") ";
  // no std::make_unique until c++14
  return std::unique_ptr<std::string>(new std::string(os.str()));
}

template <typename X, typename Y, typename Op>
DMLC_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck(X const& x, Y const& y, Op&& op) {
  if (op(x, y)) {
    return nullptr;
  }
  return ::xgboost::LogCheckFormat(x, y);
}

#define CHECK_BINARY_OP(name, op, op2, x, y)                  \
  if (auto __dmlc__log__err = ::xgboost::LogCheck(x, y, op2)) \
  ::xgboost::LogMessageFatal(__FILE__, __LINE__).stream()     \
      << "Check failed: " << #x " " #op " " #y << *__dmlc__log__err << ": "

#if defined(CHECK_LT)
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_EQ
#undef CHECK_NE
#endif  // defined(CHECK_LT)

#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, std::less<>{}, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, std::greater<>{}, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, std::less_equal<>{}, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, std::greater_equal<>{}, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, std::equal_to<>{}, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, std::not_equal_to<>{}, x, y)

#if defined(LOG)
#undef LOG
#endif  // defined(LOG)

#define LOG(severity) XGB_LOG_##severity.stream()

}  // namespace xgboost.
#endif  // XGBOOST_LOGGING_H_
