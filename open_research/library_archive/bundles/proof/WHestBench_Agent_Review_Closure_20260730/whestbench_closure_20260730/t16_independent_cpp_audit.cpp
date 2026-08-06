#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <fstream>
#include <string>
using boost::multiprecision::cpp_int;

struct V { cpp_int a,b,c; };
V mul_t(const V& v){ return {85*v.c, cpp_int(22102)*v.a+87*v.c, cpp_int(22102)*v.b-21930*v.c}; }
cpp_int ipow(cpp_int a, unsigned n){ cpp_int r=1; while(n){ if(n&1)r*=a; n>>=1; if(n)a*=a;} return r; }
cpp_int fact(unsigned n){cpp_int r=1;for(unsigned i=2;i<=n;i++)r*=i;return r;}
std::string s(const cpp_int& x){return x.convert_to<std::string>();}
int main(){
 const int N=66048, cutoff=14659;
 V prev{1,0,0}, cur{0,1,0}; cpp_int den=1;
 bool have=false; cpp_int bestn=0,bestd=1; int bestl=-1;
 for(int l=1;l<cutoff;l++){
   if(l>=6){
     cpp_int qnum=cpp_int(N-1)*cur.a-cur.b+257*cur.c;
     cpp_int rnum=-den-qnum, rden=cpp_int(N)*den;
     if(rnum>=0){std::cerr<<"nonnegative at "<<l<<"\n";return 2;}
     if(!have || rnum*bestd>bestn*rden){have=true;bestn=rnum;bestd=rden;bestl=l;}
   }
   V mt=mul_t(cur); cpp_int anum=2*l+254;
   cpp_int pm=(l==1)?cpp_int(22102):cpp_int(22102)*22102*(l+253);
   V nxt{anum*mt.a-cpp_int(l)*pm*prev.a,
         anum*mt.b-cpp_int(l)*pm*prev.b,
         anum*mt.c-cpp_int(l)*pm*prev.c};
   cpp_int denn=cpp_int(22102)*(l+254)*den;
   prev=cur;cur=nxt;den=denn;
 }
 // Exact tail inequality at cutoff: N*254!*1e6^127 < 127!*(cutoff*13951)^127
 cpp_int lhs=cpp_int(N)*fact(254)*ipow(cpp_int(1000000),127);
 cpp_int rhs=fact(127)*ipow(cpp_int(cutoff)*13951,127);
 cpp_int rhs_prev=fact(127)*ipow(cpp_int(cutoff-1)*13951,127);
 if(!(lhs<rhs) || lhs<rhs_prev){std::cerr<<"tail cutoff failure\n";return 3;}
 std::ofstream f("T16_CPP_INDEPENDENT_AUDIT.json");
 f<<"{\n  \"implementation\": \"C++17 boost::multiprecision::cpp_int\",\n";
 f<<"  \"degrees_checked\": [6, 14658],\n  \"all_strictly_negative\": true,\n";
 f<<"  \"largest_degree\": "<<bestl<<",\n";
 f<<"  \"largest_exact\": \""<<s(bestn)<<"/"<<s(bestd)<<"\",\n";
 f<<"  \"tail_cutoff\": 14659,\n  \"tail_integer_inequality_passed\": true\n}\n";
 std::cout<<"best degree "<<bestl<<" = "<<bestn<<"/"<<bestd<<"\n";
 return 0;
}
