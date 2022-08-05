#ifndef SGQUEUE_H
#define SGQUEUE_H
#include <memory>
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <condition_variable>
#include "SGCommonUtils.h"
namespace bm {
#define LOCK(name) std::lock_guard<std::mutex> guard(name##_mutex)

class SGQueueVoid {
public:
    virtual ~SGQueueVoid() {};
};

template <typename T>
class SGQueue: public Uncopiable, public SGQueueVoid
{
private:
    struct Node {
        std::shared_ptr<T> data;
        std::unique_ptr<Node> next;
    };
    std::mutex mut;
    std::unique_ptr<Node> head;
    Node* tail;
    bool joined = false;
    size_t max_nodes;
    std::atomic<size_t> num_nodes;
    std::condition_variable data_cond, slot_cond;

    std::unique_ptr<Node> popHead(){
        auto old_head = std::move(head);
        head = std::move(old_head->next);
        num_nodes.fetch_sub(1, std::memory_order_acq_rel);
        slot_cond.notify_one();
        return old_head;
    }

    std::unique_lock<std::mutex> waitForData(){
        std::unique_lock<std::mutex> ulock(mut);
        data_cond.wait(ulock, [&]{ return head.get() != tail || joined; });
        return ulock;
    }

    std::unique_ptr<Node> waitPopHead(){
        std::unique_lock<std::mutex> head_lock(waitForData());
        if (head.get() == tail) return nullptr;
        return popHead();
    }

    std::unique_ptr<Node> tryPopHead(){
        std::lock_guard<std::mutex> ulock(mut);
        if (head.get() == tail){
            return std::unique_ptr<Node>();
        }
        return popHead();
    }

public:
    SGQueue(size_t max_nodes=0): head(new Node), tail(head.get()), max_nodes(max_nodes), num_nodes(0) {}

    std::shared_ptr<T> tryPop() {
        auto oldHead = tryPopHead();
        return oldHead? oldHead->data: std::shared_ptr<T>();
    }

    bool tryPop(T& value){
        auto oldHead = tryPopHead();
        if(oldHead){
            value = std::move(*oldHead->data);
            return true;
        }
        return false;
    }

    std::shared_ptr<T> waitAndPop() {
        const auto oldHead = waitPopHead();
        if (!oldHead) return oldHead;
        return oldHead->data;
    }

    bool waitAndPop(T& value) {
        const auto oldHead = waitPopHead();
        if (!oldHead) return false;
        value = std::move(*oldHead->data);
        return true;
    }

    void join() {
        std::lock_guard<std::mutex> ulock(mut);
        joined = true;
        slot_cond.notify_all();
        data_cond.notify_all();
    }

    bool canPush() {
        return  max_nodes==0 || num_nodes<max_nodes;
    }

    void setMaxNode(size_t max){
        max_nodes = max;
    }

    void push(T new_value) {
        std::shared_ptr<T> new_data(
                    std::make_shared<T>(std::move(new_value)));
        std::unique_ptr<Node> new_node(new Node);

        std::unique_lock<std::mutex> ulock(mut);
        slot_cond.wait(ulock, [&]{ return canPush() || joined; });
        if (canPush())
        {
            tail->data = new_data;
            auto new_tail = new_node.get();
            tail->next = std::move(new_node);
            tail = new_tail;
            num_nodes.fetch_add(1, std::memory_order_acq_rel);
            data_cond.notify_one();
        }
    }

    bool empty() {
        std::lock_guard<std::mutex> ulock(mut);
        return head.get() == tail;
    }
};

template<typename T>
class SGWorkStealingQueue: public Uncopiable
{
private:
    std::deque<T> the_queue;
    mutable std::mutex queue_mutex;
public:
    SGWorkStealingQueue() {}

    void push(T data){
        LOCK(queue);
        the_queue.push_front(std::move(data));
    }

    bool empty() const {
        LOCK(queue);
        return the_queue.empty();
    }

    bool tryPop(T& res) {
        LOCK(queue);
        if(the_queue.empty()){
            return false;
        }
        res = std::move(the_queue.front());
        the_queue.pop_front();
        return true;
    }

    // steal item from back
    bool trySteal(T& res){
        LOCK(queue);
        if(the_queue.empty()){
            return false;
        }
        res = std::move(the_queue.back());
        the_queue.pop_back();
        return true;
    }

};

#undef LOCK
}
#endif // SGQUEUE_H
