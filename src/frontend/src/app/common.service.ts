import { Injectable } from '@angular/core';
import { cube, cube20, IScrambleRequest } from './rubiks/rubiks';
import { HttpService } from './http.service';

function Lock() {
  // Sets CommonService.locked to true while method is applied
  // If it already is true, the method is not applied
  return function(target: CommonService, key: string | symbol, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    descriptor.value = async function(...args: any[]) {
      if (target.locked) {
        return;
      } else {
        target.locked = true;
        const result = await original.apply(this, args);
        target.locked = false;
        return result;
      }
    };
    return descriptor;
  };
}

@Injectable({
  providedIn: 'root'
})
export class CommonService {

  locked = false;
  state: cube;
  state20: cube20;

  constructor(private httpService: HttpService) { }

  @Lock()
  public async getSolved() {
    const { state, state20 } = await this.httpService.getSolved();
    this.state = state;
    this.state20 = state20;
  }

  @Lock()
  public async act(action: number) {
    const { state, state20 } = await this.httpService.performAction({action, state20: this.state20});
    this.state = state;
    this.state20 = state20;
  }

  @Lock()
  public async scramble(depth: number) {
    const scrambleRequest: IScrambleRequest = { depth, state20: this.state20, };
    const { states, finalState20 } = await this.httpService.scramble(scrambleRequest);
    this.state20 = finalState20;
    for (const state of states) {
      const promise = new Promise<cube>((resolve, reject) => {
        setTimeout(() => resolve(state), 1000 / states.length);
      })
      this.state = await promise;
    }
  }
}
